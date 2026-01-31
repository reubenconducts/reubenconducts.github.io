# How to Use Margin Notes

You can now add marginal notes to your blog posts using the `<MarginNote>` component.

## Basic Usage

```md
This is some text with a margin note.<MarginNote number="1">This appears in the left margin!</MarginNote>
```

## Props

- `number` - The reference number or symbol (default: "*")
- `right` - Set to true to place the note on the right margin instead of left

## Examples

### Numbered notes
```md
Some text here.<MarginNote number="1">First margin note on the left</MarginNote>
More text.<MarginNote number="2">Second margin note</MarginNote>
```

### Symbol notes
```md
This uses the default asterisk.<MarginNote>A note with default symbol</MarginNote>
```

### Right-aligned notes
```md
Text with right note.<MarginNote number="1" right>This appears on the right</MarginNote>
```

## Layout

- **Left margin**: Margin notes (default placement)
- **Right side**: Table of contents navigation
- **Notes escape containers**: Margin notes will display properly even inside `:::` blocks

## Responsive Behavior

- **Large screens (≥1500px)**: Notes appear in the left margin, aligned with their reference
- **Smaller screens**: Notes appear as tooltips when hovering over the reference number

## Styling

Margin notes automatically use your site's theme colors (lavender accents) and match your serif typography.
