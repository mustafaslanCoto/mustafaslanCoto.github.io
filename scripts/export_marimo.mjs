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
  if (variant === "read") {
    const source = await readFile(sourcePath, "utf8");
    const layoutArgument =
      /^[\t ]*layout_file[\t ]*=[\t ]*(?:"[^"\n]*"|'[^'\n]*'|None)[\t ]*,[\t ]*$/m;
    let readSource = source.replace(layoutArgument, "");

    if (readSource === source && /\blayout_file\s*=/.test(source)) {
      throw new Error(
        `${sourcePath} has a non-literal or multiline layout_file; unable to create the read-only notebook variant`,
      );
    }

    if (readSource !== source) {
      temporarySourcePath = join(
        sourceDirectory,
        `.${stem}.read-export-${process.pid}.py`,
      );
      await writeFile(temporarySourcePath, readSource, { flag: "wx" });
      exportSourcePath = temporarySourcePath;
    }
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

  if (variant === "slide") {
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
