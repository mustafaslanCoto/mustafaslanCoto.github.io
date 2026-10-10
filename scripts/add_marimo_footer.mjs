import { readFile, writeFile } from "node:fs/promises";

const [htmlPath, configPath] = process.argv.slice(2);
if (!htmlPath || !configPath) {
  throw new Error("Usage: node add_marimo_footer.mjs <html-file> <footer-config>");
}

const config = JSON.parse(await readFile(configPath, "utf8"));
if (
  typeof config.text !== "string" ||
  typeof config.url !== "string" ||
  !config.text.trim() ||
  !config.url.trim()
) {
  throw new Error(`${configPath} must contain non-empty "text" and "url" strings`);
}
if (!["http:", "https:"].includes(new URL(config.url, "https://example.invalid").protocol)) {
  throw new Error(`${configPath} URL must use HTTP(S) or be a relative link`);
}

const html = await readFile(htmlPath, "utf8");
if (!html.includes("</head>") || !html.includes("</body>")) {
  throw new Error(`${htmlPath} is missing the expected HTML head or body closing tag`);
}

const safeConfig = JSON.stringify(config).replaceAll("<", "\\u003c");
const style = `
<style>
.reveal .slides > section.present {
  padding-bottom: 48px !important;
  box-sizing: border-box;
}
.reveal .marimo-footer {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 14px;
  z-index: 60;
  width: 100%;
  color: #20808D !important;
  font-size: 22px;
  line-height: normal;
  text-align: center;
  text-decoration: none !important;
  font-weight: 510;
  white-space: nowrap;
  pointer-events: auto;
}
.reveal .marimo-footer:hover,
.reveal .marimo-footer:focus {
  color: #16606A !important;
  text-decoration: none !important;
}
</style>`;
const script = `
<script>
(() => {
  const config = ${safeConfig};
  const installFooter = () => {
    const reveal = document.querySelector(".reveal");
    if (!reveal || reveal.querySelector(":scope > .marimo-footer")) return false;
    const footer = document.createElement("a");
    footer.className = "marimo-footer";
    footer.href = config.url;
    footer.textContent = config.text;
    reveal.appendChild(footer);
    return true;
  };
  if (!installFooter()) {
    const observer = new MutationObserver(() => {
      if (installFooter()) observer.disconnect();
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
  }
})();
</script>`;

const withStyle = html.replace("</head>", `${style}\n</head>`);
await writeFile(htmlPath, withStyle.replace("</body>", `${script}\n</body>`));
