<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { useSidebar } from 'vitepress/theme'
import logoBlack from '../../assets/latent-horizon/logo-black.svg'
import logoCream from '../../assets/latent-horizon/logo-cream.svg'
import { translate } from '../i18n'

const repo = 'https://github.com/latent-horizon/pytorch-world'
const license = 'https://www.apache.org/licenses/LICENSE-2.0'

const { localeIndex } = useData()
const t = (text: string) => translate(localeIndex.value, text)
// Every page exists in every locale.
const l = (link: string) => (localeIndex.value === 'root' ? link : `/${localeIndex.value}${link}`)

const columns = computed(() => [
  {
    title: t('Learn'),
    links: [
      [t('Installation'), l('/getting-started/installation')],
      [t('Quick start'), l('/getting-started/quickstart')],
      [t('Concepts'), l('/getting-started/concepts')],
      [t('Bring your own model'), l('/guides/custom-model')]
    ]
  },
  {
    title: t('Models'),
    links: [
      [t('All models'), l('/models/')],
      [t('JEPA family'), l('/models/jepa/')],
      ['LeWM', l('/models/jepa/lewm')],
      ['LeWAM', l('/models/jepa/lewam')],
      ['DINO-WM', l('/models/dino/dinowm')],
      ['RoboJEPA', l('/models/jepa/robojepa')],
      ['TD-MPC2', l('/models/tdmpc/tdmpc2')]
    ]
  },
  {
    title: t('Build'),
    links: [
      [t('Environments'), l('/environments/')],
      [t('Datasets'), l('/datasets/')],
      [t('Planning'), l('/guides/planning')],
      [t('Notebooks'), l('/examples/intro')],
      [t('API reference'), l('/api/')]
    ]
  },
  {
    title: t('Project'),
    links: [
      [t('Contributing'), l('/contributing')],
      [t('Changelog'), l('/changelog')],
      [t('Apache 2.0 license'), license],
      ['GitHub', repo]
    ]
  }
])

const { hasSidebar } = useSidebar()
const external = (url: string) => url.includes('//')
</script>

<template>
  <footer class="tw-footer" :class="{ 'has-sidebar': hasSidebar }">
    <div class="tw-footer__grid">
      <div class="tw-footer__brand">
        <a :href="withBase(l('/'))" class="tw-footer__home">
          <img class="tw-footer__logo" :src="withBase('/logo.png')" alt="" width="28" height="28" />
          <span>PyTorch World</span>
        </a>
        <p class="tw-footer__tagline">{{ t('tagline') }}</p>
        <code class="tw-footer__install">uv add torch_world</code>
      </div>
      <nav v-for="column in columns" :key="column.title" class="tw-footer__column" :aria-label="column.title">
        <p class="tw-footer__title">{{ column.title }}</p>
        <ul>
          <li v-for="[name, url] in column.links" :key="name">
            <a v-if="external(url)" :href="url" target="_blank" rel="noopener">{{ name }}</a>
            <a v-else :href="withBase(url)">{{ name }}</a>
          </li>
        </ul>
      </nav>
    </div>
    <div class="tw-footer__bar">
      <div class="tw-footer__legal">
        <span>© 2026 Latent Horizon</span>
        <a :href="license" target="_blank" rel="noopener">{{ t('Released under the Apache 2.0 license') }}</a>
      </div>
      <a class="tw-powered" href="https://latenthorizon.xyz" target="_blank" rel="noopener">
        <span class="tw-powered__label">{{ t('Powered by') }}</span>
        <img class="tw-powered__logo tw-light" :src="logoBlack" alt="Latent Horizon" width="361" height="64" />
        <img class="tw-powered__logo tw-dark" :src="logoCream" alt="Latent Horizon" width="361" height="64" />
      </a>
    </div>
  </footer>
</template>
