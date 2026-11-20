// Motion detector hot loop. Ports the key noise-rejection pieces from the
// Python ultimate_insect_detector into pure WASM:
//   RGBA -> grayscale -> 3x3 box blur -> abs frame diff
//   -> adaptive threshold (max(fixed, mean + k*std))
//   -> 3x3 morphological open (erode then dilate)
//   -> global camera-motion gate (if active ratio too high, raise thr)
//   -> 4-connectivity CC (union-find)
//   -> blob stats + shape filters (area, aspect, compactness)

use core::cell::RefCell;

#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(size);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

#[no_mangle]
pub unsafe extern "C" fn dealloc(p: *mut u8, size: usize) {
    if !p.is_null() && size > 0 {
        let _ = Vec::from_raw_parts(p, 0, size);
    }
}

thread_local! {
    static PREV: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    static DETS: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

#[no_mangle]
pub extern "C" fn reset() {
    PREV.with(|p| p.borrow_mut().clear());
}

fn find(parent: &mut [i32], mut x: i32) -> i32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

fn union(parent: &mut [i32], a: i32, b: i32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[hi as usize] = lo;
    }
}

// 3x3 box blur on grayscale (replicate borders).
fn box_blur_3x3(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    // Horizontal pass into dst as scratch via u16 sums.
    let mut tmp = vec![0u16; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let xl = if x == 0 { 0 } else { x - 1 };
            let xr = if x + 1 >= w { w - 1 } else { x + 1 };
            tmp[row + x] = src[row + xl] as u16 + src[row + x] as u16 + src[row + xr] as u16;
        }
    }
    for y in 0..h {
        let yt = if y == 0 { 0 } else { y - 1 };
        let yb = if y + 1 >= h { h - 1 } else { y + 1 };
        for x in 0..w {
            let s = tmp[yt * w + x] as u32 + tmp[y * w + x] as u32 + tmp[yb * w + x] as u32;
            dst[y * w + x] = (s / 9) as u8;
        }
    }
}

// 3x3 erode (min) on binary mask.
fn erode_3x3(mask: &[u8], out: &mut [u8], w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            let mut keep = 1u8;
            'outer: for dy in -1i32..=1 {
                let ny = y as i32 + dy;
                if ny < 0 || ny >= h as i32 { keep = 0; break 'outer; }
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    if nx < 0 || nx >= w as i32 { keep = 0; break 'outer; }
                    if mask[ny as usize * w + nx as usize] == 0 { keep = 0; break 'outer; }
                }
            }
            out[y * w + x] = keep;
        }
    }
}

// 3x3 dilate (max) on binary mask.
fn dilate_3x3(mask: &[u8], out: &mut [u8], w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            let mut hit = 0u8;
            'outer: for dy in -1i32..=1 {
                let ny = y as i32 + dy;
                if ny < 0 || ny >= h as i32 { continue; }
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    if nx < 0 || nx >= w as i32 { continue; }
                    if mask[ny as usize * w + nx as usize] != 0 { hit = 1; break 'outer; }
                }
            }
            out[y * w + x] = hit;
        }
    }
}

// Returns pointer to [count:f32, (cx,cy,w,h,area,aspect,compactness)*count].
// Each detection is 7 floats now (was 5).
#[no_mangle]
pub extern "C" fn detect(
    rgba_ptr: *const u8,
    w: u32,
    h: u32,
    fixed_threshold: u32,
    min_area: u32,
    max_area: u32,
    k_std_x100: u32,     // adaptive threshold = mean + (k_std_x100/100) * std
    camera_gate_x1000: u32, // if active ratio > gate/1000, bump threshold
    morph_open: u32,     // 0 = skip, nonzero = run 3x3 erode+dilate
) -> *const f32 {
    let w = w as usize;
    let h = h as usize;
    let n = w * h;
    let rgba = unsafe { core::slice::from_raw_parts(rgba_ptr, n * 4) };

    // RGBA -> grayscale (BT.601)
    let mut gray_raw = vec![0u8; n];
    for i in 0..n {
        let r = rgba[i * 4] as u32;
        let g = rgba[i * 4 + 1] as u32;
        let b = rgba[i * 4 + 2] as u32;
        gray_raw[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
    }

    // Blur to kill sensor noise
    let mut gray = vec![0u8; n];
    box_blur_3x3(&gray_raw, &mut gray, w, h);

    // Compute diff (abs) and simultaneously track mean/variance.
    let have_prev = PREV.with(|p| p.borrow().len() == n);
    let mut diff = vec![0u8; n];
    let mut sum: u64 = 0;
    let mut sumsq: u64 = 0;
    if have_prev {
        PREV.with(|p| {
            let prev = p.borrow();
            for i in 0..n {
                let d = (gray[i] as i32 - prev[i] as i32).unsigned_abs() as u8;
                diff[i] = d;
                sum += d as u64;
                sumsq += (d as u64) * (d as u64);
            }
        });
    }
    PREV.with(|p| *p.borrow_mut() = gray);

    // Adaptive threshold: max(fixed, mean + k * std)
    let mean = (sum as f64) / (n as f64);
    let var = (sumsq as f64) / (n as f64) - mean * mean;
    let std = if var > 0.0 { var.sqrt() } else { 0.0 };
    let k = (k_std_x100 as f64) / 100.0;
    let adaptive = mean + k * std;
    let mut thr = fixed_threshold as f64;
    if adaptive > thr { thr = adaptive; }
    if thr > 255.0 { thr = 255.0; }

    // Build binary mask + measure active ratio (camera-motion gate).
    let mut mask = vec![0u8; n];
    let mut active: u32 = 0;
    let thr_u = thr as u8;
    for i in 0..n {
        let on = (diff[i] > thr_u) as u8;
        mask[i] = on;
        active += on as u32;
    }
    // If huge fraction of pixels active -> camera is moving. Raise threshold
    // to keep only the strongest residuals (poor-man ego-motion gate).
    let gate = (camera_gate_x1000 as f64) / 1000.0;
    if (active as f64) / (n as f64) > gate && std > 0.0 {
        let thr2 = (mean + (k + 2.0) * std).min(255.0);
        let t2 = thr2 as u8;
        active = 0;
        for i in 0..n {
            let on = (diff[i] > t2) as u8;
            mask[i] = on;
            active += on as u32;
        }
    }
    let _ = active;

    // Morphological open (erode -> dilate) removes isolated noise pixels.
    if morph_open != 0 {
        let mut tmp = vec![0u8; n];
        erode_3x3(&mask, &mut tmp, w, h);
        dilate_3x3(&tmp, &mut mask, w, h);
    }

    // Connected components (4-connectivity).
    let mut labels = vec![0i32; n];
    let mut parent: Vec<i32> = vec![0];
    let mut next: i32 = 1;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if mask[i] == 0 { continue; }
            let above = if y > 0 { labels[i - w] } else { 0 };
            let left = if x > 0 { labels[i - 1] } else { 0 };
            if above != 0 && left != 0 {
                let m = above.min(left);
                labels[i] = m;
                if above != left { union(&mut parent, above, left); }
            } else if above != 0 {
                labels[i] = above;
            } else if left != 0 {
                labels[i] = left;
            } else {
                labels[i] = next;
                parent.push(next);
                next += 1;
            }
        }
    }
    for i in 0..n {
        if labels[i] != 0 { labels[i] = find(&mut parent, labels[i]); }
    }

    // Blob stats + perimeter (4-neighborhood boundary).
    let nlbl = next as usize;
    let mut sx = vec![0u64; nlbl];
    let mut sy = vec![0u64; nlbl];
    let mut area = vec![0u32; nlbl];
    let mut perim = vec![0u32; nlbl];
    let mut min_x = vec![u32::MAX; nlbl];
    let mut min_y = vec![u32::MAX; nlbl];
    let mut max_x = vec![0u32; nlbl];
    let mut max_y = vec![0u32; nlbl];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let lbl = labels[i] as usize;
            if lbl == 0 { continue; }
            sx[lbl] += x as u64;
            sy[lbl] += y as u64;
            area[lbl] += 1;
            let xu = x as u32; let yu = y as u32;
            if xu < min_x[lbl] { min_x[lbl] = xu; }
            if yu < min_y[lbl] { min_y[lbl] = yu; }
            if xu > max_x[lbl] { max_x[lbl] = xu; }
            if yu > max_y[lbl] { max_y[lbl] = yu; }
            // Count boundary edges to neighbors of different label.
            let mut p: u32 = 0;
            if x == 0 || labels[i - 1] != labels[i] { p += 1; }
            if x + 1 >= w || labels[i + 1] != labels[i] { p += 1; }
            if y == 0 || labels[i - w] != labels[i] { p += 1; }
            if y + 1 >= h || labels[i + w] != labels[i] { p += 1; }
            perim[lbl] += p;
        }
    }

    DETS.with(|d| {
        let mut dets = d.borrow_mut();
        dets.clear();
        dets.push(0.0);
        let mut count: u32 = 0;
        // Collect candidates, sort by area desc, cap at 50.
        let mut items: Vec<(u32, f32, f32, f32, f32, f32, f32)> = Vec::new();
        for lbl in 1..nlbl {
            let a = area[lbl];
            if a < min_area || a > max_area { continue; }
            let bw = (max_x[lbl] - min_x[lbl] + 1) as f32;
            let bh = (max_y[lbl] - min_y[lbl] + 1) as f32;
            // Aspect ratio filter: reject extreme slivers (edges, scanlines).
            let aspect = if bh > 0.0 { bw / bh } else { 0.0 };
            if aspect < 0.1 || aspect > 10.0 { continue; }
            // Compactness: 4*pi*area / perimeter^2. 1.0 = circle. Reject very
            // non-compact shapes (long thin edges from mis-compensation).
            let per = perim[lbl].max(1) as f32;
            let compact = (4.0 * core::f32::consts::PI * a as f32) / (per * per);
            if compact < 0.05 || compact > 1.3 { continue; }
            let cx = sx[lbl] as f32 / a as f32;
            let cy = sy[lbl] as f32 / a as f32;
            items.push((a, cx, cy, bw, bh, aspect, compact));
        }
        items.sort_by(|a, b| b.0.cmp(&a.0));
        for (a, cx, cy, bw, bh, aspect, compact) in items.into_iter().take(50) {
            dets.push(cx);
            dets.push(cy);
            dets.push(bw);
            dets.push(bh);
            dets.push(a as f32);
            dets.push(aspect);
            dets.push(compact);
            count += 1;
        }
        dets[0] = count as f32;
        dets.as_ptr()
    })
}
