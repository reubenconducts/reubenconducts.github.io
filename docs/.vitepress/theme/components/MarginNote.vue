<template>
  <span class="margin-note-wrapper">
    <span class="margin-note-ref">{{ number }}</span>
    <span class="margin-note">
      <span class="margin-note-number">{{ number }}</span>
      <slot></slot>
    </span>
  </span>
</template>

<script setup>
defineProps({
  number: {
    type: [String, Number],
    default: '*'
  }
})
</script>

<style scoped>
.margin-note-wrapper {
  position: relative;
}

.margin-note-ref {
  color: var(--vp-c-brand-1);
  font-size: 0.85em;
  vertical-align: super;
  cursor: default;
  margin: 0 0.1em;
}

.margin-note {
  position: absolute;
  left: calc(100% + 2rem);
  top: 0;
  width: 220px;
  font-size: 0.85rem;
  line-height: 1.5;
  padding: 0;
  color: var(--vp-c-text-2);
  display: none;
  pointer-events: none;
}

.margin-note-number {
  color: var(--vp-c-brand-1);
  font-weight: 600;
  margin-right: 0.3em;
}

/* Show margin notes on larger screens */
@media (min-width: 1280px) {
  .margin-note {
    display: block;
  }
}

/* On smaller screens, show as tooltip on hover */
@media (max-width: 1279px) {
  .margin-note-ref {
    cursor: pointer;
  }

  .margin-note-ref:hover + .margin-note {
    display: block;
    position: fixed;
    right: 1rem;
    top: 50%;
    transform: translateY(-50%);
    width: 200px;
    background: var(--vp-c-bg-soft);
    border: 1px solid var(--vp-c-divider);
    border-radius: 8px;
    padding: 0.75rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    z-index: 100;
    pointer-events: auto;
  }
}
</style>
