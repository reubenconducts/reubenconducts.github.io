export function highlightSass() {
  if (typeof window === 'undefined') return

  // Find all code blocks
  const codeBlocks = document.querySelectorAll('div[class*="language-"]')

  codeBlocks.forEach((block) => {
    const code = block.querySelector('code')
    if (!code) return

    // Check if this looks like SASS code
    const text = code.textContent || ''

    // More permissive detection
    if (!/\b(LDG|STG|LDS|STS|IMAD|IADD|FMA|FADD|FMUL|MOV)\b/.test(text)) return

    // Mark as SASS
    block.classList.add('language-sass')

    // Get plain text content
    let html = text

    // Escape HTML first
    html = html
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')

    // Instructions (blue)
    html = html.replace(/\b(LDG|STG|LDS|STS|LDGSTS|LDSM|LDL|STL|LD|ST|ATOM|RED|CCTL|MEMBAR|IMAD|IADD|IADD3|ISETP|IMNMX|IMUL|FLO|SHF|SHL|SHR|BFE|BFI|POPC|FMA|FADD|FMUL|FMNMX|FSET|FSETP|MUFU|RRO|HADD2|HMUL2|HFMA2|MOV|PRMT|SEL|SHFL|P2R|R2P|BRA|BRX|JMP|JMX|CALL|RET|EXIT|BRK|CONT|NOP|DEPBAR|WARPSYNC)\b/g, '<span class="sass-inst">$1</span>')

    // Modifiers
    html = html.replace(/(\.[A-Z0-9]+)\b/g, '<span class="sass-mod">$1</span>')

    // Registers (purple)
    html = html.replace(/\b(R[0-9]+|UR[0-9]+|P[0-7]|RZ|URZ|PT)\b/g, '<span class="sass-reg">$1</span>')

    // Descriptors (red)
    html = html.replace(/\b(desc|c)\b/g, '<span class="sass-desc">$1</span>')

    // Numbers
    html = html.replace(/\b(0x[0-9a-fA-F]+|[0-9]+)\b/g, '<span class="sass-num">$1</span>')

    code.innerHTML = html
  })
}
