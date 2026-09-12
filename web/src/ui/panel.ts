// Side panel cards: each can be collapsed, and one can be pinned to the top so it stays in view
// while the rest of the panel scrolls. The layout is remembered per browser.

const STORAGE_KEY = "aimbug.panel";

interface PanelState {
  collapsed: string[];
  pinned: string | null;
}

function load(ids: string[]): PanelState {
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (s && Array.isArray(s.collapsed)) {
      return {
        collapsed: s.collapsed.filter((id: unknown) => typeof id === "string" && ids.includes(id)),
        pinned: typeof s.pinned === "string" && ids.includes(s.pinned) ? s.pinned : null,
      };
    }
  } catch {
    // storage blocked or a corrupt value: start with everything open
  }
  return { collapsed: [], pinned: null };
}

function save(state: PanelState) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // not persisted, still works for this visit
  }
}

export function setupPanel(panel: HTMLElement) {
  const items = [...panel.querySelectorAll<HTMLElement>(".card[data-section]")].map((card) => {
    const h2 = card.querySelector("h2")!;
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "card-toggle";
    toggle.append(...h2.childNodes);
    const pin = document.createElement("button");
    pin.type = "button";
    pin.className = "card-pin";
    // inline pushpin icon (emoji fonts are not everywhere)
    pin.innerHTML =
      '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path fill="currentColor" d="M10.3 1.2l4.5 4.5-1.3 1.3-.9-.4-2.6 2.6.3 2.9-1.2 1.2-2.4-2.4-3.9 3.9H1.8v-.9l3.9-3.9-2.4-2.4 1.2-1.2 2.9.3 2.6-2.6-.4-.9z"/></svg>';
    h2.append(toggle, pin);
    return { id: card.dataset.section!, card, toggle, pin };
  });
  const state = load(items.map((it) => it.id));

  const apply = () => {
    for (const { id, card, toggle, pin } of items) {
      const collapsed = state.collapsed.includes(id);
      const pinned = state.pinned === id;
      card.classList.toggle("collapsed", collapsed);
      card.classList.toggle("pinned", pinned);
      toggle.setAttribute("aria-expanded", String(!collapsed));
      pin.setAttribute("aria-pressed", String(pinned));
      pin.title = pinned ? "Unpin" : "Pin to top";
      pin.setAttribute("aria-label", pinned ? "Unpin section" : "Pin section to top");
    }
  };

  for (const { id, toggle, pin } of items) {
    toggle.addEventListener("click", () => {
      state.collapsed = state.collapsed.includes(id) ? state.collapsed.filter((c) => c !== id) : [...state.collapsed, id];
      save(state);
      apply();
    });
    pin.addEventListener("click", () => {
      state.pinned = state.pinned === id ? null : id;
      // a freshly pinned section should be visible
      if (state.pinned) state.collapsed = state.collapsed.filter((c) => c !== id);
      save(state);
      apply();
    });
  }
  apply();
}
