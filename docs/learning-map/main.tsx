import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { CanvasHost } from "./canvas-shim";
import LearningMap from "./mini-sglang-learning-map.canvas";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <CanvasHost>
      <LearningMap />
    </CanvasHost>
  </StrictMode>,
);
