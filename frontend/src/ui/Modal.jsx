import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom";

const Modal = ({ children }) => {
  const [modalRoot, setModalRoot] = useState(null);

  useEffect(() => {
    let element = document.getElementById("modal-root");
    if (!element) {
      element = document.createElement("div");
      element.id = "modal-root";
      element.style.position = "relative";
      element.style.zIndex = "9999";
      document.body.appendChild(element);
    }
    setModalRoot(element);
    return () => {
      if (document.getElementById("modal-root").childElementCount === 1) {
        document.body.removeChild(element);
      }
    };
  }, []);

  if (!modalRoot) return null;

  return ReactDOM.createPortal(
    <div
      className="fixed inset-0"
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 9999,
        pointerEvents: "auto",
      }}
    >
      {children}
    </div>,
    modalRoot
  );
};

export default Modal;
