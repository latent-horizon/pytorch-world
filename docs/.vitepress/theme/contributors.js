// The home page contributors: avatars fetched from the GitHub API, so the list grows on its own. Bots are left out and
// the list is cached for the visit. If the API is unavailable (or without JavaScript), the plain link to the
// contributors graph stays in place.

const CACHE = "tw-contributors";

async function load(repo) {
  try {
    const cached = sessionStorage.getItem(`${CACHE}:${repo}`);
    if (cached) return JSON.parse(cached);
  } catch {}
  const response = await fetch(`https://api.github.com/repos/${repo}/contributors?per_page=100`);
  if (!response.ok) throw new Error(`GitHub API ${response.status}`);
  const people = (await response.json())
    .filter((c) => c.type === "User")
    .map(({ login, avatar_url, html_url }) => ({ login, avatar_url, html_url }));
  try {
    sessionStorage.setItem(`${CACHE}:${repo}`, JSON.stringify(people));
  } catch {}
  return people;
}

function render(box, people) {
  const list = document.createElement("ul");
  list.className = "tw-contributors__list";
  for (const { login, avatar_url, html_url } of people) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = html_url;
    link.target = "_blank";
    link.rel = "noopener";
    const avatar = document.createElement("img");
    avatar.src = `${avatar_url}${avatar_url.includes("?") ? "&" : "?"}s=128`;
    avatar.alt = "";
    avatar.width = avatar.height = 64;
    avatar.loading = "lazy";
    const name = document.createElement("span");
    name.textContent = login;
    link.append(avatar, name);
    item.appendChild(link);
    list.appendChild(item);
  }
  box.replaceChildren(list);
}

async function setup(box) {
  if (box.dataset.ready) return;
  box.dataset.ready = "1";
  try {
    const people = await load(box.dataset.repo);
    if (people.length) render(box, people);
  } catch {
    // keep the fallback link
  }
}

export function initContributors() {
  document.querySelectorAll(".tw-contributors[data-repo]").forEach(setup);
}
