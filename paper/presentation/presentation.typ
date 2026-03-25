#import "@preview/touying:0.6.3": *
#import themes.university: *
#import "@preview/cetz:0.4.2"
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.4.1": *
#import cosmos.clouds: *
#show: show-theorion
#let info-box(title: "Info", body) = {
  block(
    fill: blue.lighten(90%),
    stroke: (left: 4pt + blue),
    inset: 1em,
    radius: 4pt,
    width: 100%,
    stack(
      spacing: 0.5em,
      text(weight: "bold",  title),
      body
    )
  )
}

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)
#set cite(style: "apa")

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(frozen-counters: (theorem-counter,), show-notes-on-second-screen: bottom),  // freeze theorem counter for animation
  config-info(
    title: [Rethinking ABM Architecture],
    subtitle: [Entity Component Systems and the Case for Parallel Agent Interaction],
    author: [Franz Scharnreitner],
    date: datetime(year: 2026, day: 04, month:  03),
    institution: [ICAE Linz],
  ),
)

#set heading(numbering: numbly("{1}.", default: "1.1"))

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(outline(title: none, indent: 1em))

= Cars driving in circles 

== The original model


== The real world

- Real-world agents (especially cars) operate in parallel.
- Why did #cite(<hodgsonEconomicsShadowsDarwin2006>, form: "prose") choose a sequential model? #pause 
  - Obviously because it is the simple approach?? #pause
  - But why does the sequential approach lend itself so well for these models?


#pause
// Then use it like this:
#info-box(title: "The Structural Hurdle")[
  Traditional ABM frameworks structurally enforce sequential logic by coupling state and behavior.
]

#speaker-note[
  + Emphasize that while sequential is "simple" to code, it creates a massive performance bottleneck.
  + This is where ECS comes in to break the coupling.
]

== Traditional ABM layout
#cetz-canvas(length: 1.7cm, {
  import cetz.draw: *

  
  let agent-labels = ("Agent A", "Agent B", "Agent Z")
  let component-labels = ("Move",  "Check Collisions","Update Habitus", "...", "Choose next lane")
  
  let cell-width = 3.0
  let cell-height = 0.8
  let margin = 0.5 
  let ellipsis-gap = 1.5 

  // 1. Background "Model" box
  rect((-1, 1.5), (13, -6.5), fill: green.lighten(95%), stroke: green.lighten(50%), radius: 2pt)
  content((6, -5.5), text(weight: "bold", size: 20pt, fill: green.darken(40%), "Model Execution Loop"))

  // 2. Loop to draw Agents
  for (j, ag) in agent-labels.enumerate() {
    let x-offset = if j == agent-labels.len() - 1 {
      (j * (cell-width + margin)) + ellipsis-gap
    } else {
      j * (cell-width + margin)
    }
    
    if j == agent-labels.len() - 1 {
      let dots-x = x-offset - (ellipsis-gap / 2) - (margin / 2)
      content((dots-x, -2.5), text(size: 25pt, weight: "bold", fill: blue.darken(20%), [$dots$]))
    }
    
    // Agent Column Background
    rect(
      (x-offset, 0.5), (x-offset + cell-width, -(component-labels.len() + 1) * cell-height), 
      fill: blue.lighten(90%), stroke: blue.lighten(50%), radius: 2pt, name: "col-" + str(j)
    )
    
    content((x-offset + cell-width/2, 0), text(weight: "bold", size: 12pt, fill: blue.darken(20%), ag))
    
    for (i, cl) in component-labels.enumerate() {
      let y-pos = -(i + 1) * cell-height
      rect(
        (x-offset + 0.2, y-pos + 0.3), (x-offset + cell-width - 0.2, y-pos - 0.3), 
        fill: white, stroke: gray.lighten(50%), radius: 1pt, name: "step-" + str(j) + "-" + str(i)
      )
      content((x-offset + cell-width/2, y-pos), text(size: 9pt, style: "italic", cl))
    }
  }

  // 3. The Flow Arrows (Appearing after a pause)
  
    set-style(stroke: (dash: "solid", thickness: 1.5pt, paint: orange.darken(10%), cap: "round"), mark: (fill: orange.darken(10%), end: ">"))
  
    bezier((1.0, -5.0), (3.5 + 1.5, 0.3), (5.5, -6.5), (3.-3 + 1.5, 1.5))
    bezier((5.0, -5.0), (5.5 + 1.5, 0.3), (11.5, -6.5), (5.-3 + 1.5, 1.5))
})

== What if?

#cetz-canvas(length: 1.6cm, {
  import cetz.draw: *
  
  let agent-labels = ("Agent A", "Agent B", "Agent Z")
  let component-labels = ("Move",  "Check Collisions","Update Habitus", "...", "Choose next lane")
  
  let cell-width = 3.0
  let cell-height = 0.8
  let margin = 0.5 
  let ellipsis-gap = 1.5 

  // 1. Background "Model" box
  rect((-1, 1.5), (13, -6.5), fill: green.lighten(95%), stroke: green.lighten(50%), radius: 2pt)
  content((6, -5.5), text(weight: "bold", size: 20pt, fill: green.darken(40%), "Model Execution Loop"))

  // 2. Loop to draw Agents
  for (j, ag) in agent-labels.enumerate() {
    let x-offset = if j == agent-labels.len() - 1 {
      (j * (cell-width + margin)) + ellipsis-gap
    } else {
      j * (cell-width + margin)
    }
    
    if j == agent-labels.len() - 1 {
      let dots-x = x-offset - (ellipsis-gap / 2) - (margin / 2)
      content((dots-x, -2.5), text(size: 25pt, weight: "bold", fill: blue.darken(20%), [$dots$]))
    }
    
    // Agent Column Background
    rect(
      (x-offset, 0.5), (x-offset + cell-width, -(component-labels.len() + 1) * cell-height), 
      fill: blue.lighten(90%), stroke: blue.lighten(50%), radius: 2pt, name: "col-" + str(j)
    )
    
    content((x-offset + cell-width/2, 0), text(weight: "bold", size: 12pt, fill: blue.darken(20%), ag))
    
    for (i, cl) in component-labels.enumerate() {
      let y-pos = -(i + 1) * cell-height
      rect(
        (x-offset + 0.2, y-pos + 0.3), (x-offset + cell-width - 0.2, y-pos - 0.3), 
        fill: white, stroke: gray.lighten(50%), radius: 1pt, name: "step-" + str(j) + "-" + str(i)
      )
      content((x-offset + cell-width/2, y-pos), text(size: 9pt, style: "italic", cl))
    }
  }


  rect((-0.5,-0.3), (12,-1.3), fill: none, stroke: red, radius: 2pt)
  (pause,)
  rect((-0.5,-3.6), (12,-4.5), fill: none, stroke: red, radius: 2pt)
})



== Complex Animation

At subslide #touying-fn-wrapper((self: none) => str(self.subslide)), we can

use #uncover("2-")[`#uncover` function] for reserving space,

use #only("2-")[`#only` function] for not reserving space,

#alternatives[call `#only` multiple times \u{2717}][use `#alternatives` function #sym.checkmark] for choosing one of the alternatives.


== Callback Style Animation

#slide(
  repeat: 3,
  self => [
    #let (uncover, only, alternatives) = utils.methods(self)

    At subslide #self.subslide, we can

    use #uncover("2-")[`#uncover` function] for reserving space,

    use #only("2-")[`#only` function] for not reserving space,

    #alternatives[call `#only` multiple times \u{2717}][use `#alternatives` function #sym.checkmark] for choosing one of the alternatives.
  ],
)


== Math Equation Animation

Equation with `pause`:

$
  f(x) &= pause x^2 + 2x + 1 \
  &= pause (x + 1)^2 \
$

#meanwhile

Here, #pause we have the expression of $f(x)$.

#pause

By factorizing, we can obtain this result.


== CeTZ Animation

CeTZ Animation in Touying:

#cetz-canvas({
  import cetz.draw: *

  rect((0, 0), (5, 5))

  (pause,)

  rect((0, 0), (1, 1))
  rect((1, 1), (2, 2))
  rect((2, 2), (3, 3))

  (pause,)

  line((0, 0), (2.5, 2.5), name: "line")
})


== Fletcher Animation

Fletcher Animation in Touying:

#fletcher-diagram(
  node-stroke: .1em,
  node-fill: gradient.radial(blue.lighten(80%), blue, center: (30%, 20%), radius: 80%),
  spacing: 4em,
  edge((-1, 0), "r", "-|>", `open(path)`, label-pos: 0, label-side: center),
  node((0, 0), `reading`, radius: 2em),
  edge((0, 0), (0, 0), `read()`, "--|>", bend: 130deg),
  pause,
  edge(`read()`, "-|>"),
  node((1, 0), `eof`, radius: 2em),
  pause,
  edge(`close()`, "-|>"),
  node((2, 0), `closed`, radius: 2em, extrude: (-2.5, 0)),
  edge((0, 0), (2, 0), `close()`, "-|>", bend: -40deg),
)


= Theorems

== Prime numbers

#definition[
  A natural number is called a #highlight[_prime number_] if it is greater
  than 1 and cannot be written as the product of two smaller natural numbers.
]
#example[
  The numbers $2$, $3$, and $17$ are prime.
  @cor_largest_prime shows that this list is not exhaustive!
]

#theorem(title: "Euclid")[
  There are infinitely many primes.
]
#pagebreak(weak: true)
#proof[
  Suppose to the contrary that $p_1, p_2, dots, p_n$ is a finite enumeration
  of all primes. Set $P = p_1 p_2 dots p_n$. Since $P + 1$ is not in our list,
  it cannot be prime. Thus, some prime factor $p_j$ divides $P + 1$. Since
  $p_j$ also divides $P$, it must divide the difference $(P + 1) - P = 1$, a
  contradiction.
]

#corollary[
  There is no largest prime number.
] <cor_largest_prime>
#corollary[
  There are infinitely many composite numbers.
]

#theorem[
  There are arbitrarily long stretches of composite numbers.
]

#proof[
  For any $n > 2$, consider $
    n! + 2, quad n! + 3, quad ..., quad n! + n
  $
]


= Others

== Side-by-side

#slide(composer: (1fr, 1fr))[
  First column.
][
  Second column.
]


== Multiple Pages

#lorem(200)


#show: appendix

= Appendix

== Appendix

#bibliography("../Econ.bib")
Please pay attention to the current slide number.
