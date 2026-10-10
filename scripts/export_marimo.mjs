import { readFile, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const [sourceArgument, variant, outputArgument] = process.argv.slice(2);
if (!sourceArgument || !["read", "slide", "edit"].includes(variant) || !outputArgument) {
  throw new Error(
    "Usage: node export_marimo.mjs <notebook.py> <read|slide|edit> <output.html>",
  );
}

const sourcePath = resolve(sourceArgument);
const outputPath = resolve(outputArgument);
const sourceDirectory = dirname(sourcePath);
const sourceName = basename(sourcePath);
const stem = sourceName.slice(0, -".py".length);
const footerConfigPath = join(sourceDirectory, `${stem}.footer.json`);
let exportSourcePath = sourcePath;
let temporarySourcePath;

try {
  const source = await readFile(sourcePath, "utf8");
  const layoutArgument =
    /^[\t ]*layout_file[\t ]*=[\t ]*(?:"[^"\n]*"|'[^'\n]*'|None)[\t ]*,[\t ]*$/m;
  const hasLayoutArgument = /\blayout_file\s*=/.test(source);
  let exportSource = source;

  const isSlide = variant === "slide";
  if (isSlide && !hasLayoutArgument) {
    const appConstructor = /^[\t ]*(\w+)[\t ]*=[\t ]*(?:marimo|mo)\.App\(/m;
    const appMatch = source.match(appConstructor);
    if (!appMatch) {
      throw new Error(`${sourcePath} does not have a marimo.App constructor to configure for slides`);
    }
    const appName = appMatch[1];
    const cellDecorator = new RegExp(`^[\\t ]*@${appName}\\.cell\\b`, "gm");
    const cellCount = [...source.matchAll(cellDecorator)].length;
    if (cellCount === 0) {
      throw new Error(`${sourcePath} does not contain any @${appName}.cell definitions`);
    }

    const slideLayout = {
      type: "slides",
      data: { cells: Array.from({ length: cellCount }, () => ({})), deck: {} },
    };
    const layoutDataUri = `data:application/json;base64,${Buffer.from(
      JSON.stringify(slideLayout),
    ).toString("base64")}`;
    exportSource = source.replace(
      appConstructor,
      (match) => `${match}\n    layout_file=${JSON.stringify(layoutDataUri)},`,
    );
  } else if (variant !== "slide" && hasLayoutArgument) {
    exportSource = source.replace(layoutArgument, "");
    if (exportSource === source) {
      throw new Error(
        `${sourcePath} has a non-literal or multiline layout_file; unable to create the ${variant} notebook variant`,
      );
    }
  }

  if (exportSource !== source) {
    temporarySourcePath = join(
      sourceDirectory,
      `.${stem}.${variant}-export-${process.pid}.py`,
    );
    await writeFile(temporarySourcePath, exportSource, { flag: "wx" });
    exportSourcePath = temporarySourcePath;
  }

  const exportMode = variant === "edit" ? "edit" : "run";
  const exportResult = spawnSync(
    "marimo",
    [
      "export",
      "html-wasm",
      exportSourcePath,
      "--output",
      outputPath,
      "--mode",
      exportMode,
      "--single-file",
      "--force",
      "--no-sandbox",
    ],
    { cwd: sourceDirectory, encoding: "utf8", stdio: "inherit" },
  );
  if (exportResult.error) throw exportResult.error;
  if (exportResult.status !== 0) {
    throw new Error(`marimo export failed for ${sourcePath} (${variant} mode)`);
  }

  if (temporarySourcePath) {
    const html = await readFile(outputPath, "utf8");
    const title = stem.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    await writeFile(
      outputPath,
      html.replace(/<title>[\s\S]*?<\/title>/, `<title>${title}</title>`),
    );
  }

  if (isSlide) {
    if (await fileExists(footerConfigPath)) {
      const footer = spawnSync(
        process.execPath,
        [
          fileURLToPath(new URL("./add_marimo_footer.mjs", import.meta.url)),
          outputPath,
          footerConfigPath,
        ],
        { encoding: "utf8", stdio: "inherit" },
      );
      if (footer.error) throw footer.error;
      if (footer.status !== 0) {
        throw new Error(`Could not add the configured footer to ${outputPath}`);
      }
    }
  }
} finally {
  if (temporarySourcePath) await rm(temporarySourcePath, { force: true });
}

async function fileExists(path) {
  try {
    await readFile(path);
    return true;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}
