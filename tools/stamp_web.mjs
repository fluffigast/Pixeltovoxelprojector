// stamp_web.mjs — Give web/index.html and its modules one shared version stamp.
//
// GitHub Pages lets browsers reuse every file for ten minutes without asking.
// After a deploy a browser can therefore combine a new index.html with old
// modules (or the other way round), and the page breaks until the old copies
// expire. With a stamp in every module URL, a page always loads the modules it
// was published with, and app.js can tell when a newer version is out.
//
// The stamp is a hash of index.html (without the stamp itself) and all modules,
// so it changes exactly when the page changes.
//
//   node tools/stamp_web.mjs          write the stamp into web/index.html
//   node tools/stamp_web.mjs --check  exit 1 if the stamp is out of date
//
// Run it before committing a change to web/index.html or web/*.js.

import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const WEB = fileURLToPath(new URL('../web/', import.meta.url));
const INDEX = join(WEB, 'index.html');
const NOT_MODULES = new Set(['server.js']);          // runs in Node, not in the page
const ENTRY = 'app.js';

const BLOCK = /<!-- VERSION-STAMP:BEGIN[^>]*-->[\s\S]*?<!-- VERSION-STAMP:END -->/;
const ENTRY_TAG = /<script type="module" src="app\.js(\?v=[0-9a-f]*)?"><\/script>/;

const html = readFileSync(INDEX, 'utf8');
if (!BLOCK.test(html) || !ENTRY_TAG.test(html)) {
  console.error('web/index.html: version-stamp markers or the app.js script tag are missing');
  process.exit(2);
}

const modules = readdirSync(WEB).filter(f => f.endsWith('.js') && !NOT_MODULES.has(f)).sort();

// Hash the page with the stamp taken out, so stamping does not change the hash.
const neutral = html
  .replace(BLOCK, '<!-- VERSION-STAMP -->')
  .replace(ENTRY_TAG, '<script type="module" src="app.js"></script>');
const hash = createHash('sha256').update(neutral);
for (const f of modules) hash.update('\0' + f + '\0').update(readFileSync(join(WEB, f)));
const stamp = hash.digest('hex').slice(0, 10);

// Relative keys are resolved against the page URL, so `import './x.js'` from any
// module in web/ is redirected to the stamped URL.
const imports = {};
for (const f of modules) if (f !== ENTRY) imports['./' + f] = './' + f + '?v=' + stamp;

const stamped = html
  .replace(BLOCK, '<!-- VERSION-STAMP:BEGIN written by tools/stamp_web.mjs, do not edit by hand -->\n' +
    '  <script type="importmap">' + JSON.stringify({ imports }) + '</script>\n' +
    '  <!-- VERSION-STAMP:END -->')
  .replace(ENTRY_TAG, '<script type="module" src="app.js?v=' + stamp + '"></script>');

if (process.argv.includes('--check')) {
  if (stamped !== html) {
    console.error('web/index.html: version stamp is out of date, run: node tools/stamp_web.mjs');
    process.exit(1);
  }
  console.log('version stamp up to date: ' + stamp);
} else {
  if (stamped !== html) writeFileSync(INDEX, stamped);
  console.log((stamped !== html ? 'stamped ' : 'already stamped ') + stamp + ' (' + modules.length + ' modules)');
}
