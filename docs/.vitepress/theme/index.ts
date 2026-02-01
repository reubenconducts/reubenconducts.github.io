import DefaultTheme from 'vitepress/theme'
import './custom.css'
import { highlightSass, highlightPTX } from './sassHighlight'
import { onMounted, watch } from 'vue'
import { useRoute } from 'vitepress'

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()

    const runHighlighters = () => {
      highlightSass()
      highlightPTX()
    }

    onMounted(() => {
      runHighlighters()
    })

    watch(() => route.path, () => {
      setTimeout(runHighlighters, 100)
    })
  }
}
