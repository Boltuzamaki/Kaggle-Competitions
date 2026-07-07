import { expect, test } from "@playwright/test";

async function expectStatus(page, text) {
  await expect(page.locator(".status")).toContainText(text, { timeout: 10000 });
}

test("project controls and graph workflow are wired end to end", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Task 010")).toBeVisible();
  await expect(page.getByRole("region", { name: "Export payload inspector" })).toBeVisible();
  await expect(page.getByTestId("payload-node-count")).toHaveText("2");
  await expect(page.getByTestId("payload-edge-count")).toHaveText("1");
  await expect(page.getByRole("heading", { name: "BEST TEMPLATE" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Load Graph" })).toBeVisible();
  await expect(page.getByRole("link", { name: /Task 010 ONNX/ })).toHaveAttribute("href", "/best/onnx/task010.onnx");
  await expect(page.getByRole("link", { name: /Full best zip/ })).toHaveAttribute("href", "/best/submission-best.zip");
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: /Task 010 ONNX/ }).click();
  await downloadPromise;
  await expect(page.getByTestId("payload-last-download")).toContainText("task010.onnx");
  await page.getByRole("button", { name: "Load Graph" }).click();
  await expectStatus(page, "Loaded best ONNX graph");
  await expect(page.getByTestId("payload-node-count")).toHaveText("86");

  await page.getByRole("button", { name: "Load Selected" }).click();
  await expectStatus(page, "Loaded project baseline-cast-equal");
  await expect(page.locator(".nodeId", { hasText: "equal_1" })).toBeVisible();
  await expect(page.getByTestId("payload-node-count")).toHaveText("5");
  await page.locator(".nodeId", { hasText: "equal_1" }).click();
  await expect(page.getByTestId("payload-selected")).toContainText("equal_1");

  await page.getByRole("button", { name: "Compile" }).click();
  await expectStatus(page, "Compile passed");

  await page.getByRole("button", { name: /Run/ }).click();
  await expect(page.getByText("RUN OUTPUT")).toBeVisible();
  await expectStatus(page, "Run produced");
  await expect(page.getByTestId("payload-last-run")).toContainText("passed");

  await page.getByRole("button", { name: "New" }).click();
  await expectStatus(page, "New project neurogolf-task010");
  await expect(page.getByTestId("payload-node-count")).toHaveText("2");
  await expect(page.locator(".nodeId", { hasText: "input_1" })).toBeVisible();
  await expect(page.locator(".nodeId", { hasText: "output_1" })).toBeVisible();
  await expect(page.locator(".nodeId", { hasText: "equal_1" })).toHaveCount(0);

  await page.locator(".quickAdd select").selectOption("Constant");
  await page.locator(".quickAdd button").click();
  await expect(page.locator(".nodeId", { hasText: "constant_1" })).toBeVisible();
  await expect(page.getByTestId("payload-node-count")).toHaveText("3");

  await page.getByLabel("Project name").fill("e2e-project");
  await page.getByRole("button", { name: /Save/ }).click();
  await expectStatus(page, "Saved project e2e-project");

  await page.getByRole("button", { name: "New" }).click();
  await expect(page.locator(".nodeId", { hasText: "constant_1" })).toHaveCount(0);

  await page.getByLabel("Saved project").selectOption("e2e-project");
  await page.getByRole("button", { name: "Load Selected" }).click();
  await expectStatus(page, "Loaded project e2e-project");
  await expect(page.locator(".nodeId", { hasText: "constant_1" })).toBeVisible();

  await page.locator(".nodeId", { hasText: "constant_1" }).click();
  await page.keyboard.press("Delete");
  await expect(page.locator(".nodeId", { hasText: "constant_1" })).toHaveCount(0);
  await expect(page.getByTestId("payload-node-count")).toHaveText("2");

  await page.getByRole("button", { name: "Train" }).click();
  await expectStatus(page, "train");
  await page.getByRole("button", { name: "Test" }).click();
  await expectStatus(page, "test");
  await page.getByRole("button", { name: "Extra 10" }).click();
  await expectStatus(page, "extra");
});
