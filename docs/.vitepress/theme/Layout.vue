<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import { useData, useRoute } from 'vitepress'
import { nextTick, onMounted, watch } from 'vue'
import Footer from './Footer.vue'
import { translate } from '../i18n'
import { initPixels } from './pixels.js'
import { initFilm } from './film.js'

const { frontmatter, localeIndex } = useData()
const t = (text: string) => translate(localeIndex.value, text)
const route = useRoute()
const repo = 'https://github.com/latent-horizon/pytorch-world'

// The home page hero and film are plain markup; enhance them after every navigation.
const enhance = () => nextTick(() => (initPixels(), initFilm()))
onMounted(enhance)
watch(() => route.path, enhance)
</script>

<template>
  <DefaultTheme.Layout>
    <template #doc-before>
      <div v-if="frontmatter.notebook" class="tw-notebook">
        <a class="tw-notebook__button" :href="`${repo}/blob/main/${frontmatter.notebook}`" target="_blank" rel="noopener">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path
              fill="currentColor"
              d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2.2c-3.3.7-4-1.4-4-1.4-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1 0-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-6 0-1.2.5-2.3 1.3-3.1-.2-.4-.6-1.6 0-3.2 0 0 1-.3 3.4 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.6 1.6.2 2.8.1 3.2.8.8 1.3 1.9 1.3 3.2 0 4.6-2.8 5.6-5.5 5.9.5.4.9 1.1.9 2.2v3.3c0 .3.1.7.8.6A12 12 0 0 0 12 .3"
            />
          </svg>
          {{ t('View on GitHub') }}
        </a>
        <a class="tw-notebook__button" :href="`${repo}/raw/main/${frontmatter.notebook}`" download>
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M12 3v12m0 0-5-5m5 5 5-5M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"
            />
          </svg>
          {{ t('Download notebook') }}
        </a>
      </div>
    </template>
    <template #layout-bottom>
      <Footer />
    </template>
  </DefaultTheme.Layout>
</template>
