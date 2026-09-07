import "@testing-library/jest-dom/vitest";

// jsdom has no PointerEvent; the resize handle listens for pointer events, so a MouseEvent stands in.
if (typeof window !== "undefined" && typeof window.PointerEvent === "undefined") {
  class PointerEventPolyfill extends MouseEvent {
    readonly pointerId: number;
    readonly pointerType: string;
    constructor(type: string, init: PointerEventInit = {}) {
      super(type, init);
      this.pointerId = init.pointerId ?? 0;
      this.pointerType = init.pointerType ?? "mouse";
    }
  }
  Object.defineProperty(window, "PointerEvent", { writable: true, value: PointerEventPolyfill });
}
