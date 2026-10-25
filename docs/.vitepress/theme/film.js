// The home page film: a poster with one frosted play button. The first press starts playback and hands over to the
// native controls; when the film ends it returns to its poster. Without JavaScript the native controls stay.

const PLAY = '<svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true"><path d="M8 5.5v13l11-6.5z" fill="currentColor"/></svg>';

function setup(film) {
  const video = film.querySelector("video");
  if (!video || film.dataset.ready) return;
  film.dataset.ready = "1";
  video.controls = false;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "tw-film__play";
  button.setAttribute("aria-label", video.title || "Play");
  button.innerHTML = PLAY;
  button.addEventListener("click", () => {
    film.classList.add("tw-film--playing");
    video.controls = true;
    video.play();
  });
  video.addEventListener("ended", () => {
    film.classList.remove("tw-film--playing");
    video.controls = false;
    video.load();
  });
  film.appendChild(button);
}

export function initFilm() {
  document.querySelectorAll(".tw-film").forEach(setup);
}
