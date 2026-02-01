import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'
import container from 'markdown-it-container'

const customElements = [
  'mjx-container',
  'mjx-assistive-mml',
  'math',
  'maction',
  'maligngroup',
  'malignmark',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mi',
  'mlongdiv',
  'mmultiscripts',
  'mn',
  'mo',
  'mover',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'ms',
  'mscarries',
  'mscarry',
  'mscarries',
  'msgroup',
  'mstack',
  'mlongdiv',
  'msline',
  'mstack',
  'mspace',
  'msqrt',
  'msrow',
  'mstack',
  'mstack',
  'mstyle',
  'msub',
  'msup',
  'msubsup',
  'mtable',
  'mtd',
  'mtext',
  'mtr',
  'munder',
  'munderover',
  'semantics',
  'math',
  'mi',
  'mn',
  'mo',
  'ms',
  'mspace',
  'mtext',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'msqrt',
  'mstyle',
  'mmultiscripts',
  'mover',
  'mprescripts',
  'msub',
  'msubsup',
  'msup',
  'munder',
  'munderover',
  'none',
  'maligngroup',
  'malignmark',
  'mtable',
  'mtd',
  'mtr',
  'mlongdiv',
  'mscarries',
  'mscarry',
  'msgroup',
  'msline',
  'msrow',
  'mstack',
  'maction',
  'semantics',
  'annotation',
  'annotation-xml'
]

export default defineConfig({
  title: 'Reuben Stern',
  description: 'Kernel Engineer, Conductor, Mathematician',
  base: '/',
  appearance: false,

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'About', link: '/about' },
      { text: 'Resources', link: '/resources'},
    ],

    outline: {
      level: [2, 3],
      label: 'On this page'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/reubenconducts' }
    ],

    footer: {
      message: 'Built with VitePress',
      copyright: 'Copyright © Reuben Stern'
    }
  },

  markdown: {
    theme: {
      light: 'catppuccin-latte',
      dark: 'catppuccin-latte'
    },
    codeTransformers: [
      {
        name: 'sass-highlighter',
        preprocess(code, options) {
          if (options.lang === 'sass' || options.lang === 'sass-asm') {
            // Tell Shiki to treat it as plaintext so we can style it ourselves
            options.lang = 'text'
          }
          return code
        }
      }
    ],
    config: (md) => {
      md.use(mathjax3)

      // Register custom containers
      const containerTypes = ['definition', 'theorem', 'lemma', 'proposition', 'corollary', 'example', 'aside']

      containerTypes.forEach(type => {
        md.use(container, type, {
          render: (tokens: any[], idx: number) => {
            const token = tokens[idx]
            const info = token.info.trim().slice(type.length).trim()
            if (token.nesting === 1) {
              return `<div class="custom-block ${type}"><p class="custom-block-title">${info}</p>\n`
            } else {
              return '</div>\n'
            }
          }
        })
      })
    }
  },

  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => customElements.includes(tag)
      }
    }
  },

  head: [
    [
      'script',
      {},
      `
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        },
        svg: {
          fontCache: 'global'
        }
      };
      `
    ],
    [
      'script',
      {
        async: true,
        src: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js'
      }
    ]
  ]
})
