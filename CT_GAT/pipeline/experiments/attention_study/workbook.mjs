import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error("Usage: node workbook.mjs snapshot.json results.xlsx");
}
const snapshot = JSON.parse(await fs.readFile(inputPath, "utf8"));
const workbook = Workbook.create();
const sheets = [
  ["Attention review", snapshot.summary],
  ["Validation runs", snapshot.validation],
  ["Held-out runs", snapshot.held_out],
];
for (const [name, table] of sheets) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const values = [table.columns, ...table.rows];
  const range = sheet.getRangeByIndexes(0, 0, values.length, table.columns.length);
  range.values = values;
  range.format.font = { name: "Arial", size: 10, color: "#172334" };
  range.format.rowHeight = 21;
  range.format.columnWidth = 17;
  range.format.verticalAlignment = "center";
  const header = sheet.getRangeByIndexes(0, 0, 1, table.columns.length);
  header.format = {
    fill: "#253B55",
    font: { name: "Arial", size: 10, color: "#FFFFFF", bold: true },
    rowHeight: name === "Attention review" ? 26 : 42,
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  if (name === "Attention review") {
    sheet.getRange("A1:A15").format.columnWidth = 42;
    sheet.getRange("B1:B15").format.columnWidth = 88;
    sheet.getRange("B2:B15").format.wrapText = true;
    sheet.getRange("A2:B15").format.rowHeight = 33;
    sheet.getRange("B14").setNumberFormat("yyyy-mm-dd hh:mm");
    sheet.tabColor = "#253B55";
  } else {
    sheet.freezePanes.freezeRows(1);
    sheet.freezePanes.freezeColumns(3);
    sheet.getRangeByIndexes(0, 0, values.length, 1).format.columnWidth = 24;
    for (let col = 0; col < table.columns.length; col++) {
      const key = table.columns[col];
      const body = sheet.getRangeByIndexes(1, col, table.rows.length, 1);
      if (["accuracy", "auc", "pr_auc", "precision", "detection_rate", "specificity",
           "fpr", "macro_f1", "validation_auc", "validation_macro_f1"].includes(key)) {
        body.setNumberFormat("0.0%");
      } else if (key.includes("dropout") && key !== "dropout_control") {
        body.setNumberFormat("0.0");
      } else if (["lr", "weight_decay", "optimizer_eps"].includes(key)) {
        body.setNumberFormat("0.000E+00");
      } else if (["epoch_seconds_mean", "peak_gpu_mb"].includes(key)) {
        body.setNumberFormat("0.0");
      }
      if (key.includes("digest") || key.includes("fingerprint")) {
        sheet.getRangeByIndexes(0, col, values.length, 1).format.columnWidth = 66;
      } else if (key === "source_record") {
        sheet.getRangeByIndexes(0, col, values.length, 1).format.columnWidth = 90;
      } else if (["effective_gat_heads", "class_weight_rule", "dropout_control"].includes(key)) {
        sheet.getRangeByIndexes(0, col, values.length, 1).format.columnWidth = 31;
      } else if (["representation", "pooling"].includes(key)) {
        sheet.getRangeByIndexes(0, col, values.length, 1).format.columnWidth = 25;
      }
    }
    let lastColumn = "";
    for (let index = table.columns.length; index > 0; index = Math.floor((index - 1) / 26)) {
      lastColumn = String.fromCharCode(65 + (index - 1) % 26) + lastColumn;
    }
    sheet.tables.add(`A1:${lastColumn}${values.length}`, true,
                     name === "Validation runs" ? "ValidationRuns" : "HeldOutRuns");
  }
}
workbook.recalculate();
console.log((await workbook.inspect({
  kind: "sheet", include: "id,name", maxChars: 1600,
})).ndjson);
await fs.mkdir(path.dirname(outputPath), { recursive: true });
for (const [name] of sheets) {
  const preview = await workbook.render({
    sheetName: name, range: name === "Attention review" ? "A1:B15" : "A1:J10",
    scale: 1.5, format: "png",
  });
  await fs.writeFile(path.join(path.dirname(outputPath), name.replaceAll(" ", "_") + ".png"),
                    new Uint8Array(await preview.arrayBuffer()));
}
const exported = await SpreadsheetFile.exportXlsx(workbook);
const temporary = outputPath + ".tmp.xlsx";
await exported.save(temporary);
await fs.rename(temporary, outputPath);
console.log(JSON.stringify({ outputPath, validationRows: snapshot.validation.rows.length,
                            heldOutRows: snapshot.held_out.rows.length }));
