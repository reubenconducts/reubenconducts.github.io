---
layout: post
title: Using LaTeX Math in Jekyll Blog Posts
categories: [Jekyll, Markdown]
mathjax: true
---

# Using LaTeX Math in Jekyll Blog Posts

This post demonstrates how to use LaTeX math expressions in your Jekyll blog posts. We've enabled MathJax support to render mathematical expressions directly in your browser.

## Inline Math Expressions

You can include inline math expressions by wrapping them in single dollar signs. For example, the formula for the quadratic formula $ax^2 + bx + c = 0$ can be solved using $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.

## Block Math Expressions

For more complex equations or to display them on their own line, use double dollar signs:

$$
\begin{align}
E &= mc^2 \\
m &= \frac{m_0}{\sqrt{1-\frac{v^2}{c^2}}} \\
F &= G\frac{m_1 m_2}{r^2}
\end{align}
$$

## Examples of Mathematical Notation

Here are some more examples of mathematical notation:

1. **Integrals**:
   $$\int_{a}^{b} f(x) \, dx$$

2. **Limits**:
   $$\lim_{x \to \infty} \frac{1}{x} = 0$$

3. **Matrices**:
   $$
   A = \begin{pmatrix}
   a_{11} & a_{12} & a_{13} \\
   a_{21} & a_{22} & a_{23} \\
   a_{31} & a_{32} & a_{33}
   \end{pmatrix}
   $$

4. **Fractions and Binomials**:
   $$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

5. **Greek Letters**:
   $$\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \eta, \theta$$

## Conclusion

With MathJax enabled, you can now include complex mathematical expressions in your blog posts. This is particularly useful for technical or scientific content. 