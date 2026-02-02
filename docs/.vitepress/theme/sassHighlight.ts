function highlightPTXLine(line: string): string {
  // Check for comment
  const commentIdx = line.indexOf('//')
  let codePart = commentIdx >= 0 ? line.substring(0, commentIdx) : line
  let commentPart = commentIdx >= 0 ? line.substring(commentIdx) : ''

  // Escape HTML in code part
  codePart = codePart
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')

  // Directives
  codePart = codePart.replace(/(\.(version|target|address_size|visible|entry|func|param|reg|local|shared|global|const|align|b8|b16|b32|b64|u8|u16|u32|u64|s8|s16|s32|s64|f16|f32|f64|pred|v2|v4))\b/g, '<span class="ptx-directive">$1</span>')

  // Instructions
  codePart = codePart.replace(/\b(ld|st|mov|add|sub|mul|mad|div|rem|abs|neg|min|max|cvt|set|setp|selp|and|or|xor|not|shl|shr|bra|brx|call|ret|exit|bar|atom|red|vote|shfl|mad24|mul24|sad|fma|rcp|sqrt|rsqrt|sin|cos|lg2|ex2)\b/g, '<span class="ptx-inst">$1</span>')

  // Registers
  codePart = codePart.replace(/(%[a-zA-Z_][a-zA-Z0-9_]*|%[rpf]d?[0-9]+)/g, '<span class="ptx-reg">$1</span>')

  // Numbers
  codePart = codePart.replace(/\b(0[xX][0-9a-fA-F]+|[0-9]+\.?[0-9]*[fF]?)\b/g, '<span class="ptx-num">$1</span>')

  // Labels
  if (/^[a-zA-Z_][a-zA-Z0-9_]*:/.test(codePart)) {
    codePart = codePart.replace(/^([a-zA-Z_][a-zA-Z0-9_]*):/, '<span class="ptx-label">$1:</span>')
  }

  // Escape and highlight comment
  if (commentPart) {
    commentPart = commentPart
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
    commentPart = '<span class="ptx-comment">' + commentPart + '</span>'
  }

  return codePart + commentPart
}

export function highlightPTX() {
  if (typeof window === 'undefined') return

  // Find all code elements directly
  const codeElements = document.querySelectorAll('code')

  codeElements.forEach((code) => {
    // Skip if already processed
    if (code.classList.contains('ptx-highlighted')) return

    const text = code.textContent || ''

    // Detect PTX code
    if (!/\.(version|target|address_size|visible|entry|func|param|reg|local)\b/.test(text) &&
        !/\b(ld|st|mov|add|sub|mul|mad|cvt|setp|bra|ret)\b/.test(text)) return

    // Find parent language block
    const block = code.closest('div[class*="language-"]')
    if (block) {
      block.classList.add('language-ptx')
    }

    // Mark as processed
    code.classList.add('ptx-highlighted')

    const lines = text.split('\n')
    const html = lines.map(highlightPTXLine).join('\n')

    code.innerHTML = html
  })
}

export function highlightSass() {
  if (typeof window === 'undefined') return

  // Find all code elements directly
  const codeElements = document.querySelectorAll('code')

  codeElements.forEach((code) => {
    // Skip if already processed
    if (code.classList.contains('sass-highlighted')) return

    // Find parent language block
    const block = code.closest('div[class*="language-"]')

    // Check if explicitly marked as sass
    const isExplicitSass = block?.className.includes('language-sass')

    // Check if this looks like SASS code
    const text = code.textContent || ''

    // More permissive detection - check for common SASS instructions
    const hasSassInstructions = /\b(LDG|STG|LDS|STS|LDGSTS|LDSM|LDL|STL|LD|ST|LDC|LDCU|ATOM|RED|CCTL|MEMBAR|IMAD|IADD|IADD3|ISETP|UISETP|IMNMX|IMUL|FLO|SHF|SHL|SHR|USHF|BFE|BFI|POPC|FMA|FADD|FMUL|FMNMX|FSET|FSETP|MUFU|RRO|HADD2|HMUL2|HFMA2|MOV|PRMT|SEL|USEL|SHFL|P2R|R2P|R2UR|S2R|BRA|BRX|JMP|JMX|CALL|RET|EXIT|BRK|CONT|NOP|DEPBAR|WARPSYNC)\b/.test(text)

    if (!isExplicitSass && !hasSassInstructions) return

    if (block) {
      block.classList.add('language-sass')
    }

    // Mark as processed
    code.classList.add('sass-highlighted')

    // Get plain text content
    let html = text

    // Escape HTML first
    html = html
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')

    // Instructions (blue)
    html = html.replace(/\b(LDG|STG|LDS|STS|LDGSTS|LDSM|LDL|STL|LD|ST|LDC|LDCU|ATOM|RED|CCTL|MEMBAR|IMAD|IADD|IADD3|ISETP|UISETP|IMNMX|IMUL|FLO|SHF|SHL|SHR|USHF|BFE|BFI|POPC|FMA|FFMA|FADD|FMUL|FMNMX|FSET|FSETP|FSEL|MUFU|RRO|HADD2|HMUL2|HFMA2|HMMA|DMMA|MOV|PRMT|SEL|USEL|SHFL|P2R|R2P|R2UR|S2R|BRA|BRX|JMP|JMX|CALL|RET|EXIT|BRK|CONT|NOP|DEPBAR|WARPSYNC)\b/g, '<span class="sass-inst">$1</span>')

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
