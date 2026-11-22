# Dump a video as raw RGBA frames for tools/motion_eval.js.
# Usage: venv/bin/python -I tools/dump_frames.py <video.mp4> <out_prefix>
#   writes <out_prefix>.rgba and <out_prefix>.meta.json
import cv2, json, sys
vid, out = sys.argv[1], sys.argv[2]
c = cv2.VideoCapture(vid); n = 0
with open(out + '.rgba', 'wb') as fh:
    while True:
        ok, fr = c.read()
        if not ok: break
        fh.write(cv2.cvtColor(fr, cv2.COLOR_BGR2RGBA).tobytes()); n += 1; H, W = fr.shape[:2]
json.dump({'w': W, 'h': H, 'n': n}, open(out + '.meta.json', 'w'))
print('dumped', n, 'frames', W, 'x', H)
