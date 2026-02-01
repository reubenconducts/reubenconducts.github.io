import DefaultTheme from 'vitepress/theme'
import './custom.css'
import { highlightSass } from './sassHighlight'
import { onMounted, watch } from 'vue'
import { useRoute } from 'vitepress'

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()

    onMounted(() => {
      highlightSass()
    })

    watch(() => route.path, () => {
      setTimeout(highlightSass, 100)
    })
  }
}
