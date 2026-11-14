import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { chromium } from 'playwright';

const viewer = (await readFile(new URL('./viewer.html', import.meta.url), 'utf8'))
  .replaceAll('https://cdn.jsdelivr.net/npm/three@0.187.0/', '/three/');
const three = new URL('./node_modules/three/', import.meta.url);
const headScan = await readFile(new URL('./assets/head-scan.glb', import.meta.url));
const origin = 'http://haircut-renderer.invalid';

async function loadPage(page, state) {
  // Route the scene data directly into the private browser page. There is no HTTP listener.
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) return route.abort();
    const path = url.pathname;
    if (path === '/') return route.fulfill({ contentType: 'text/html', body: viewer });
    if (path === '/state') return route.fulfill({ json: state });
    if (path === '/assets/head-scan.glb') return route.fulfill({ contentType: 'model/gltf-binary', body: headScan });
    if (!path.startsWith('/three/') || !path.endsWith('.js')) return route.fulfill({ status: 404 });
    const file = new URL(path.slice('/three/'.length), three);
    if (!file.href.startsWith(three.href)) return route.fulfill({ status: 404 });
    try {
      const script = await readFile(file);
      return route.fulfill({ contentType: 'text/javascript', body: script });
    } catch {
      return route.fulfill({ status: 404 });
    }
  });
  await page.goto(`${origin}/?capture`);
  await page.waitForFunction(() => typeof globalThis.captureCameras === 'function', { timeout: 15000 });
}

let browser, page, pageError;
try {
  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of input) {
    try {
      const message = JSON.parse(line);
      if (!browser) {
        browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
        page = await browser.newPage({ viewport: { width: 512, height: 384 } });
        page.on('pageerror', error => { pageError = error.message; });
        await loadPage(page, message);
        process.stdout.write(JSON.stringify({ ready: true }) + '\n');
      } else {
        const pictures = await page.evaluate(frame => globalThis.captureCameras(frame), message);
        process.stdout.write(JSON.stringify(pictures) + '\n');
      }
    } catch (error) {
      process.stdout.write(JSON.stringify({ error: pageError ?? error.message }) + '\n');
      break;
    }
  }
} finally {
  await browser?.close();
}
