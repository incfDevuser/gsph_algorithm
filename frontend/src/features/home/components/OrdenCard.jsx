import React from "react";
import { MapPin, Clock, Package, ChevronRight, UserCircle } from "lucide-react";

const OrdenCard = ({ order }) => {
  const timeWindow =
    order.id.charCodeAt(0) % 2 === 0 ? "08:30–10:30" : "13:00–15:00";
  const getOrderStatus = () => {
    const lastChar = order.id.slice(-1);
    if ("123".includes(lastChar)) return { text: "Pendiente", color: "red" };
    if ("456".includes(lastChar)) return { text: "Retirado", color: "yellow" };
    if ("789".includes(lastChar)) return { text: "Programado", color: "blue" };
    return { text: "Pendiente", color: "red" };
  };

  const status = getOrderStatus();
  const getStatusClasses = (color) => {
    switch (color) {
      case "red":
        return "bg-red-100 text-red-700";
      case "yellow":
        return "bg-yellow-100 text-yellow-700";
      case "blue":
        return "bg-blue-100 text-blue-700";
      default:
        return "bg-gray-100 text-gray-700";
    }
  };

  return (
    <article className="border-b border-neutral-200 px-4 py-3 transition hover:bg-neutral-50">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <UserCircle className="h-5 w-5 text-neutral-400" />
          <span className="text-sm font-medium text-neutral-900">
            Pedido #{order.id.substring(0, 5)}
          </span>
        </div>
        <div
          className={`rounded-full px-2 py-0.5 text-xs font-medium ${getStatusClasses(
            status.color
          )}`}
        >
          {status.text}
        </div>
      </div>
      <div className="mt-2">
        <ul className="space-y-2 text-sm">
          <li className="flex items-center gap-2 text-neutral-600">
            <div className="flex h-5 w-5 items-center justify-center">
              <MapPin className="h-4 w-4 text-neutral-500" />
            </div>
            <span className="flex-1 truncate">
              {order.lat.toFixed(4)}, {order.lng.toFixed(4)}
            </span>
          </li>
          <li className="flex items-center gap-2 text-neutral-600">
            <div className="flex h-5 w-5 items-center justify-center">
              <Clock className="h-4 w-4 text-neutral-500" />
            </div>
            <span className="flex-1">{timeWindow}</span>
          </li>
          <li className="flex items-center gap-2 text-neutral-600">
            <div className="flex h-5 w-5 items-center justify-center">
              <Package className="h-4 w-4 text-neutral-500" />
            </div>
            <span className="flex-1">Paquete estandar</span>
          </li>
        </ul>
      </div>
      <div className="mt-3 flex items-center justify-between">
        <div className="text-xs text-neutral-500">
          Priority: {(order.id.charCodeAt(1) % 3) + 1}/3
        </div>
      </div>
    </article>
  );
};

export default OrdenCard;
