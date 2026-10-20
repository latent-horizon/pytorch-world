// Carousels for the home page environment gallery: one horizontally scrolling row per section, with minimal arrows
// that page by the visible width and hide at either end. Without JavaScript the row still scrolls by hand.

const CHEVRON = (points) =>
  `<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><polyline points="${points}" fill="none" ` +
  `stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

function setup(carousel) {
  const track = carousel.querySelector(":scope > p");
  if (!track || carousel.dataset.ready) return;
  carousel.dataset.ready = "1";
  const arrow = (direction) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `tw-carousel__arrow tw-carousel__arrow--${direction > 0 ? "next" : "prev"}`;
    b.setAttribute("aria-label", direction > 0 ? "Next environments" : "Previous environments");
    b.innerHTML = CHEVRON(direction > 0 ? "9 5 16 12 9 19" : "15 5 8 12 15 19");
    b.addEventListener("click", () => track.scrollBy({ left: direction * track.clientWidth * 0.85, behavior: "smooth" }));
    carousel.appendChild(b);
    return b;
  };
  const prev = arrow(-1);
  const next = arrow(1);
  const update = () => {
    const start = track.scrollLeft <= 1;
    const end = track.scrollLeft + track.clientWidth >= track.scrollWidth - 1;
    prev.hidden = start;
    next.hidden = end;
    carousel.classList.toggle("tw-carousel--more-before", !start);
    carousel.classList.toggle("tw-carousel--more-after", !end);
  };
  track.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
}

export function initCarousels() {
  document.querySelectorAll(".tw-carousel").forEach(setup);
}
