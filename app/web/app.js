// Front-end for the captioning dashboard.
// Sends the chosen image to the inference server and shows the returned caption.

const ENDPOINT = "http://localhost:8000";

const fileInput = document.getElementById("file");
const preview = document.getElementById("preview");
const hint = document.getElementById("hint");
const result = document.getElementById("result");
const captionEl = document.getElementById("caption");
document.getElementById("endpoint").textContent = ENDPOINT;

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  // Preview
  const reader = new FileReader();
  reader.onloadend = () => {
    preview.src = reader.result;
    preview.style.display = "block";
    hint.style.display = "none";
  };
  reader.readAsDataURL(file);

  // Request caption
  result.hidden = false;
  captionEl.textContent = "Generating…";
  try {
    const response = await fetch(ENDPOINT, {
      method: "POST",
      body: file,
      headers: { "Content-Type": "image/png" },
      cache: "no-cache",
    });
    captionEl.textContent = response.ok
      ? await response.text()
      : `Server error (${response.status})`;
  } catch (err) {
    captionEl.textContent =
      "Could not reach the model server. Is it running on " + ENDPOINT + "?";
  }
});
