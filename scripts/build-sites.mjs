import { promises as fs } from "node:fs";
import path from "node:path";

const projectRoot = path.resolve(import.meta.dirname, "..");
const sourceRoot = path.join(projectRoot, "_site");
const outputRoot = path.join(projectRoot, "dist");
const workerPath = path.join(outputRoot, "server", "index.js");

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
};

async function listFiles(directory) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const files = await Promise.all(
    entries.map(async (entry) => {
      const absolute = path.join(directory, entry.name);
      return entry.isDirectory() ? listFiles(absolute) : [absolute];
    }),
  );
  return files.flat();
}

await fs.access(path.join(sourceRoot, "index.html"));
const files = await listFiles(sourceRoot);
const assets = [];

for (const file of files.sort()) {
  const relative = `/${path.relative(sourceRoot, file).split(path.sep).join("/")}`;
  const extension = path.extname(file).toLowerCase();
  const contentType = mimeTypes[extension] ?? "application/octet-stream";
  const body = await fs.readFile(file);
  assets.push([relative, [contentType, body.toString("base64")]]);
}

const worker = `const assets = new Map(${JSON.stringify(assets)});

function decode(body) {
  return Uint8Array.from(atob(body), (character) => character.charCodeAt(0));
}

function resolvePath(pathname) {
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return null;
  }

  if (decoded.endsWith("/")) return decoded + "index.html";
  if (assets.has(decoded)) return decoded;
  if (assets.has(decoded + ".html")) return decoded + ".html";
  if (assets.has(decoded + "/index.html")) return decoded + "/index.html";
  return null;
}

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const resolved = resolvePath(url.pathname);
    const asset = resolved ? assets.get(resolved) : assets.get("/404.html");

    if (!asset) {
      return new Response("Not found", { status: 404 });
    }

    const [contentType, encodedBody] = asset;
    const headers = new Headers({
      "Content-Type": contentType,
      "X-Content-Type-Options": "nosniff",
      "Referrer-Policy": "strict-origin-when-cross-origin",
      "Cache-Control": contentType.startsWith("text/html")
        ? "public, max-age=0, must-revalidate"
        : "public, max-age=86400",
    });

    return new Response(request.method === "HEAD" ? null : decode(encodedBody), {
      status: resolved ? 200 : 404,
      headers,
    });
  },
};
`;

await fs.rm(outputRoot, { recursive: true, force: true });
await fs.mkdir(path.dirname(workerPath), { recursive: true });
await fs.writeFile(workerPath, worker);

console.log(`Packaged ${assets.length} static files for Sites.`);
