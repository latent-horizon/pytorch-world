import { defineConfig, type DefaultTheme } from 'vitepress'
import { LOCALES, translate } from './i18n'
import { torchWorld } from './markdown'

const repo = 'https://github.com/latent-horizon/pytorch-world'

type Item = DefaultTheme.SidebarItem

// The nav and sidebars of one locale ('root' is English). Every page exists in every locale (prepare.py copies the
// English page where there is no translation), so every link takes the locale prefix.
function site(key: string) {
  const prefix = key === 'root' ? '' : `/${key}`
  const t = (text: string) => translate(key, text)
  const l = (link: string) => prefix + link
  const page = (text: string, link: string): Item => ({ text: t(text), link })
  const group = (text: string, items: Item[], collapsed?: boolean): Item => ({ text: t(text), items, collapsed })
  const envs = (dir: string, names: string[]): Item[] =>
    names.map((name) => page(name, l(`/examples/envs/${dir}/${name.toLowerCase()}_v0`)))

  const gettingStarted: Item[] = [
    group('Get started', [
      page('Installation', l('/getting-started/installation')),
      page('Quick start', l('/getting-started/quickstart')),
      page('Concepts', l('/getting-started/concepts'))
    ])
  ]

  const guides: Item[] = [
    group('Guides', [page('Overview', l('/guides/'))]),
    group('Bring your own', [
      page('Model', l('/guides/custom-model')),
      page('Environment', l('/guides/custom-environment')),
      page('Dataset', l('/guides/custom-dataset')),
      page('Planner', l('/guides/custom-planner'))
    ]),
    group('Train and evaluate', [
      page('Training', l('/guides/training')),
      page('Masking', l('/guides/masking')),
      page('Planning', l('/guides/planning')),
      page('Metrics', l('/guides/metrics')),
      page('Figures', l('/guides/plots'))
    ]),
    group('Benchmarks', [
      page('Benchmark experiments', l('/guides/benchmark-environments')),
      page('Pixel worlds', l('/guides/environments'))
    ])
  ]

  const models: Item[] = [
    group('Models', [page('Overview', l('/models/'))]),
    group('JEPA', [
      page('Overview', l('/models/jepa/')),
      group('Action', [
        page('LeWM', l('/models/jepa/lewm')),
        page('PLDM', l('/models/jepa/pldm')),
        page('LpWM', l('/models/jepa/lpwm')),
        page('Flow-JEPA', l('/models/jepa/flowjepa')),
        page('Variational JEPA', l('/models/jepa/varjepa')),
        page('HWM', l('/models/jepa/hwm')),
        page('H-JEPA', l('/models/jepa/hjepa')),
        page('LeWAM', l('/models/jepa/lewam')),
        page('RoboJEPA', l('/models/jepa/robojepa'))
      ]),
      group('Vision', [
        page('I-JEPA', l('/models/jepa/ijepa')),
        page('LeJEPA', l('/models/jepa/lejepa')),
        page('V-JEPA', l('/models/jepa/vjepa')),
        page('V-JEPA 2.1', l('/models/jepa/vjepa21')),
        page('LeVJEPA', l('/models/jepa/levjepa'))
      ]),
      group('Vision-language', [page('VL-JEPA', l('/models/jepa/vljepa'))])
    ]),
    group('DINO', [
      page('Overview', l('/models/dino/')),
      page('DINOv2', l('/models/dino/dinov2')),
      page('DINO-WM', l('/models/dino/dinowm'))
    ]),
    group('TD-MPC', [page('Overview', l('/models/tdmpc/')), page('TD-MPC2', l('/models/tdmpc/tdmpc2'))]),
    group('World action models', [page('Overview', l('/models/wam/')), page('DreamZero', l('/models/wam/dreamzero'))])
  ]

  const environments: Item[] = [
    group('Environments', [page('Overview', l('/environments/'))]),
    group('Planning puzzles', envs('puzzles', ['Checkpoints', 'IceMaze', 'KeyDoor', 'Maze', 'Sokoban'])),
    group(
      'Control',
      envs('control', ['BallPlate', 'CartpoleSwingup', 'DoubleCartpole', 'ReactionWheel', 'NewtonAcrobot', 'NewtonPendulum'])
    ),
    group(
      'Navigation and locomotion',
      envs('navigation', ['Macaw', 'PointMass', 'DifferentialParking', 'HurdleHopper', 'PayloadTransport', 'PlanarQuadrotor'])
    ),
    group(
      'Manipulation',
      envs('manipulation', [
        'BlockStack', 'BoxGate', 'DoorOpen', 'DrawerOpen', 'ObstacleArm', 'PegInsertion', 'PickPlace', 'PuckPush',
        'PuckSlide', 'PushT', 'RobotCube', 'RobotCubeSO101', 'SpatialArm', 'FrankaCubeLift', 'FrankaObstacleReach',
        'FrankaTablePush'
      ]),
      true
    ),
    group('Benchmark suites', [
      page('OGBench', l('/examples/envs/ogbench')),
      page('DMControl', l('/examples/envs/dmcontrol')),
      page('Meta-World', l('/examples/envs/metaworld')),
      page('ManiSkill', l('/examples/envs/maniskill')),
      page('MyoSuite', l('/examples/envs/myosuite')),
      page('RoboDojo', l('/examples/envs/robodojo')),
      page('RoboCasa', l('/examples/envs/robocasa'))
    ])
  ]

  const datasets: Item[] = [
    group('Datasets', [
      page('Overview', l('/datasets/')),
      page('Trajectories', l('/datasets/trajectories')),
      page('MedMNIST', l('/datasets/medmnist')),
      page('Medical VQA', l('/datasets/medical-vqa'))
    ])
  ]

  const notebooks: Item[] = [
    group('Notebooks', [page('Tour', l('/examples/intro')), page('Custom model', l('/examples/custom_model'))]),
    group('Action', [
      page('LeWM', l('/examples/action/lewm')),
      page('PLDM', l('/examples/action/pldm')),
      page('LpWM', l('/examples/action/lpwm')),
      page('Flow-JEPA', l('/examples/action/flowjepa')),
      page('Variational JEPA', l('/examples/action/varjepa')),
      page('HWM', l('/examples/action/hwm')),
      page('H-JEPA', l('/examples/action/hjepa')),
      page('LeWAM', l('/examples/action/lewam')),
      page('DINO-WM', l('/examples/action/dinowm')),
      page('RoboJEPA', l('/examples/action/robojepa')),
      page('TD-MPC2', l('/examples/action/tdmpc2')),
      page('DreamZero', l('/examples/action/dreamzero'))
    ]),
    group('Vision', [
      page('I-JEPA', l('/examples/vision/ijepa')),
      page('LeJEPA', l('/examples/vision/lejepa')),
      page('V-JEPA', l('/examples/vision/vjepa')),
      page('V-JEPA 2.1', l('/examples/vision/vjepa21')),
      page('LeVJEPA', l('/examples/vision/levjepa')),
      page('DINOv2 vs LeVJEPA', l('/examples/vision/dinov2'))
    ]),
    group('Vision-language', [page('VL-JEPA', l('/examples/vqa/vljepa'))]),
    group('Environments', [page('Custom model', l('/examples/envs/custom_model_three_envs'))])
  ]

  const modules = ['nn', 'masks', 'losses', 'models', 'planning', 'envs', 'datasets', 'evaluation', 'metrics', 'utils', 'plots', 'hub']
  const api: Item[] = [
    group('API', [page('Overview', l('/api/')), ...modules.map((m) => page(`torch_world.${m}`, l(`/api/${m}`)))])
  ]

  const nav: DefaultTheme.NavItem[] = [
    { text: t('Get started'), link: l('/getting-started/installation'), activeMatch: `^${prefix}/getting-started/` },
    { text: t('Guides'), link: l('/guides/'), activeMatch: `^${prefix}/guides/` },
    { text: t('Models'), link: l('/models/'), activeMatch: `^${prefix}/models/` },
    {
      text: t('Environments'),
      link: l('/environments/'),
      activeMatch: `^${prefix}/(environments/|examples/envs/(?!custom_model))`
    },
    { text: t('Datasets'), link: l('/datasets/'), activeMatch: `^${prefix}/datasets/` },
    { text: t('Notebooks'), link: l('/examples/intro'), activeMatch: `^${prefix}/examples/(?!envs/(?!custom_model))` },
    { text: 'API', link: l('/api/'), activeMatch: `^${prefix}/api/` },
    {
      text: t('Project'),
      items: [
        { text: t('Contributing'), link: l('/contributing') },
        { text: t('Changelog'), link: l('/changelog') }
      ]
    }
  ]

  // The longer prefix wins a tie in depth, so the one environment notebook listed under Notebooks comes first.
  const sidebar: DefaultTheme.SidebarMulti = {
    [l('/examples/envs/custom_model_three_envs')]: notebooks,
    [l('/getting-started/')]: gettingStarted,
    [l('/guides/')]: guides,
    [l('/models/')]: models,
    [l('/environments/')]: environments,
    [l('/examples/envs/')]: environments,
    [l('/datasets/')]: datasets,
    [l('/examples/')]: notebooks,
    [l('/api/')]: api
  }
  return { nav, sidebar }
}

const en = site('root')

function localeTheme(key: string): DefaultTheme.Config {
  const { theme } = LOCALES[key]
  const { nav, sidebar } = site(key)
  return {
    nav,
    sidebar,
    outline: { level: [2, 3], label: theme.outline },
    docFooter: { prev: theme.prev, next: theme.next },
    darkModeSwitchLabel: theme.appearance,
    lightModeSwitchTitle: theme.toLight,
    darkModeSwitchTitle: theme.toDark,
    sidebarMenuLabel: theme.menu,
    returnToTopLabel: theme.top,
    langMenuLabel: theme.language,
    notFound: { title: theme.notFound[0], quote: theme.notFound[1], linkText: theme.notFound[2], link: `/${key}/` }
  }
}

function searchLocale(key: string) {
  const [button, noResults, select, navigate, close, reset, back, details] = LOCALES[key].theme.search
  return {
    translations: {
      button: { buttonText: button, buttonAriaLabel: button },
      modal: {
        noResultsText: noResults,
        resetButtonTitle: reset,
        backButtonTitle: back,
        displayDetails: details,
        footer: { selectText: select, navigateText: navigate, closeText: close }
      }
    }
  }
}

export default defineConfig({
  title: 'PyTorch World',
  description: 'World models library for PyTorch',
  base: '/pytorch-world/',
  cleanUrls: true,
  lastUpdated: false,
  // Pages not yet translated, notebooks and the API reference among them, are English copies that prepare.py writes
  // to docs/_fallback/<locale>/.
  rewrites: { '_fallback/:rest*': ':rest*' },
  locales: {
    root: { label: 'English', lang: 'en-US' },
    ...Object.fromEntries(
      Object.entries(LOCALES).map(([key, locale]) => [
        key,
        { label: locale.label, lang: locale.lang, description: locale.description, themeConfig: localeTheme(key) }
      ])
    )
  },
  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/pytorch-world/logo-light.png' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap' }],
    ['meta', { property: 'og:title', content: 'PyTorch World' }],
    ['meta', { property: 'og:description', content: 'World models library for PyTorch' }],
    ['meta', { property: 'og:url', content: 'https://latent-horizon.github.io/pytorch-world/' }]
  ],
  // The local search index is one lazily loaded chunk per locale of about 700 kB.
  vite: { build: { chunkSizeWarningLimit: 1000 } },
  markdown: {
    math: true,
    config: (md) => md.use(torchWorld)
  },
  themeConfig: {
    logo: '/logo-white.png',
    nav: en.nav,
    sidebar: en.sidebar,
    socialLinks: [{ icon: 'github', link: repo }],
    search: {
      provider: 'local',
      options: {
        locales: Object.fromEntries(Object.keys(LOCALES).map((key) => [key, searchLocale(key)])),
        // Search the API reference by its docs, not by the folded source of every object.
        _render(src, env, md) {
          const html = md.render(src, env)
          if (env.frontmatter?.search === false) return ''
          return html.replace(/<details class="api-source">[\s\S]*?<\/details>/g, '')
        }
      }
    },
    outline: { level: [2, 3], label: 'On this page' }
  }
})
