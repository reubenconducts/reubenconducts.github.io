import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'
import container from 'markdown-it-container'
import { getMathJaxMacrosString, mathjaxMacros } from './mathjax-macros.mts'

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
      {
        text: 'Blog',
        items: [
          { text: 'All Posts', link: '/archive' },
          { text: 'Lost in the SASS', link: '/posts/lost-in-the-sass/about-lits' },
          { text: 'FlexAttention Backward', link: '/posts/flex-bwd' }
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'Glossary', link: '/glossary' },
          { text: 'Resources', link: '/resources' }
        ]
      },
      { text: 'About', link: '/about' },
      { text: 'Music', link: '/music' },
      { text: 'Math',  link: '/math'  },
    ],

    outline: {
      level: [2, 4],
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
    lineNumbers: true,
    theme: {
      light: 'catppuccin-latte',
      dark: 'catppuccin-latte'
    },
    codeTransformers: [
      {
        name: 'custom-asm-highlighter',
        preprocess(code, options) {
          if (options.lang === 'sass' || options.lang === 'sass-asm' || options.lang === 'ptx') {
            // Tell Shiki to treat it as plaintext so we can style it ourselves
            options.lang = 'text'
          }
          return code
        }
      }
    ],
    config: (md) => {
      md.use(mathjax3, {
        tex: {
          macros: mathjaxMacros,
          packages: {'[+]': ['ams', 'boldsymbol', 'newcommand', 'configmacros', 'action', 'unicode']}
          // Available packages: base, ams, amscd, bbox, boldsymbol, braket, bussproofs,
          // cancel, cases, centernot, color, colortbl, empheq, enclose, extpfeil,
          // gensymb, html, mathtools, mhchem, newcommand, noerrors, noundefined,
          // physics, textcomp, textmacros, unicode, verb, configmacros, tagformat, etc.
        }
      })

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
      'link',
      { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }
    ],
    [
      'script',
      {},
      `
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
          packages: {'[+]': ['ams', 'newcommand', 'configmacros', 'action', 'unicode']},
          macros: ${getMathJaxMacrosString()}
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
