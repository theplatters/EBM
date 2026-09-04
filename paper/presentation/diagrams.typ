#import "@preview/touying:0.6.3": pause, touying-reducer
#import "@preview/cetz:0.4.2"
#import "@preview/fletcher:0.5.8" as fletcher: edge, node

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#let traditional-abm-layout = cetz-canvas(length: 1.7cm, {
  import cetz.draw: *


  let agent-labels = ("Agent A", "Agent B", "Agent Z")
  let component-labels = ("Move", "Check Collisions", "Update Habitus", "...", "Choose next lane")

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
      (x-offset, 0.5),
      (x-offset + cell-width, -(component-labels.len() + 1) * cell-height),
      fill: blue.lighten(90%),
      stroke: blue.lighten(50%),
      radius: 2pt,
      name: "col-" + str(j),
    )

    content((x-offset + cell-width / 2, 0), text(weight: "bold", size: 12pt, fill: blue.darken(20%), ag))

    for (i, cl) in component-labels.enumerate() {
      let y-pos = -(i + 1) * cell-height
      rect(
        (x-offset + 0.2, y-pos + 0.3),
        (x-offset + cell-width - 0.2, y-pos - 0.3),
        fill: white,
        stroke: gray.lighten(50%),
        radius: 1pt,
        name: "step-" + str(j) + "-" + str(i),
      )
      content((x-offset + cell-width / 2, y-pos), text(size: 9pt, style: "italic", cl))
    }
  }

  // 3. The Flow Arrows (Appearing after a pause)

  set-style(
    stroke: (dash: "solid", thickness: 1.5pt, paint: orange.darken(10%), cap: "round"),
    mark: (fill: orange.darken(10%), end: ">"),
  )

  bezier((1.0, -5.0), (3.5 + 1.5, 0.3), (5.5, -6.5), (3. - 3 + 1.5, 1.5))
  bezier((5.0, -5.0), (5.5 + 1.5, 0.3), (11.5, -6.5), (5. - 3 + 1.5, 1.5))
})

#let what-if-layout = cetz-canvas(length: 1.6cm, {
  import cetz.draw: *

  let agent-labels = ("Agent A", "Agent B", "Agent Z")
  let component-labels = ("Move", "Check Collisions", "Update Habitus", "...", "Choose next lane")

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
      (x-offset, 0.5),
      (x-offset + cell-width, -(component-labels.len() + 1) * cell-height),
      fill: blue.lighten(90%),
      stroke: blue.lighten(50%),
      radius: 2pt,
      name: "col-" + str(j),
    )

    content((x-offset + cell-width / 2, 0), text(weight: "bold", size: 12pt, fill: blue.darken(20%), ag))

    for (i, cl) in component-labels.enumerate() {
      let y-pos = -(i + 1) * cell-height
      rect(
        (x-offset + 0.2, y-pos + 0.3),
        (x-offset + cell-width - 0.2, y-pos - 0.3),
        fill: white,
        stroke: gray.lighten(50%),
        radius: 1pt,
        name: "step-" + str(j) + "-" + str(i),
      )
      content((x-offset + cell-width / 2, y-pos), text(size: 9pt, style: "italic", cl))
    }
  }


  rect((-0.5, -0.3), (12, -1.3), fill: none, stroke: red, radius: 2pt)
  (pause,)
  rect((-0.5, -3.6), (12, -4.5), fill: none, stroke: red, radius: 2pt)
})

#let classic-abm-data-layout = cetz-canvas(length: 1.6cm, {
  import cetz.draw: *

  // -----------------------------
  // Configuration
  // -----------------------------
  let agent-labels = ("Agent A", "Agent B", "Agent Z")
  let data-labels = ("Position", "LR", "Habitus", "Parameters")

  let cell-width = 3.0
  let cell-height = 0.8
  let margin = 0.5
  let data-offset = 0.4
  let ellipsis-gap = 1.5

  let agent-top = 0.5
  let agent-bottom = -(data-labels.len() + 3) * cell-height

  let data-title-y = -0.5
  let method-title-y = -4.1
  let method-box-y = -4.7

  let resource-x = 12.2

  // Array container padding
  let array-pad-x = 0.35
  let array-pad-top = 0.55
  let array-pad-bottom = 0.75

  // -----------------------------
  // Styles
  // -----------------------------
  let world-fill = green.lighten(95%)
  let world-stroke = green.lighten(50%)
  let world-text = green.darken(40%)

  let panel-fill = blue.lighten(90%)
  let panel-stroke = blue.lighten(50%)
  let panel-text = blue.darken(20%)

  let box-fill = white
  let box-stroke = gray.lighten(50%)

  let array-stroke = blue.darken(10%)
  let array-fill = blue.lighten(96%)

  // -----------------------------
  // Helpers
  // -----------------------------
  let draw-entry-box(x, y, w, label) = {
    rect(
      (x + 0.2, y + 0.3),
      (x + w - 0.2, y - 0.3),
      fill: box-fill,
      stroke: box-stroke,
      radius: 1pt,
    )
    content((x + w / 2, y), text(size: 9pt, style: "italic", label))
  }

  let draw-agent-column(x, label) = {
    rect(
      (x, agent-top),
      (x + cell-width, agent-bottom),
      fill: panel-fill,
      stroke: panel-stroke,
      radius: 2pt,
    )

    content(
      (x + cell-width / 2, 0),
      text(weight: "bold", size: 12pt, fill: panel-text, label),
    )

    content(
      (x + 0.5, data-title-y),
      text(weight: "bold", size: 12pt, fill: panel-text, "Data:"),
    )

    for (i, item) in data-labels.enumerate() {
      let y = -(i + 1) * cell-height - data-offset
      draw-entry-box(x, y, cell-width, item)
    }

    content(
      (x + 0.7, method-title-y),
      text(weight: "bold", size: 12pt, fill: panel-text, "Methods:"),
    )
    draw-entry-box(x, method-box-y, cell-width, "update")
  }

  let draw-resource-panel(x, y-top, y-bottom, title, subtitle) = {
    rect(
      (x, y-top),
      (x + cell-width, y-bottom),
      fill: panel-fill,
      stroke: panel-stroke,
      radius: 2pt,
    )

    let center-y = (y-top + y-bottom) / 2
    content(
      (x + cell-width / 2, center-y + 0.5),
      text(weight: "bold", size: 12pt, fill: panel-text, title),
    )
    content(
      (x + cell-width / 2, center-y - 0.5),
      text(weight: "bold", size: 12pt, fill: panel-text, subtitle),
    )
  }

  let agent-x(j) = if j == agent-labels.len() - 1 {
    j * (cell-width + margin) + ellipsis-gap
  } else {
    j * (cell-width + margin)
  }

  // -----------------------------
  // World background
  // -----------------------------
  rect(
    (-1, 1.5),
    (16, -6.8),
    fill: world-fill,
    stroke: world-stroke,
    radius: 2pt,
  )
  content(
    (6, -6.2),
    text(weight: "bold", size: 20pt, fill: world-text, "World"),
  )

  // -----------------------------
  // Agents array container
  // -----------------------------
  let first-agent-x = agent-x(0)
  let last-agent-x = agent-x(agent-labels.len() - 1)

  let array-left = first-agent-x - array-pad-x
  let array-right = last-agent-x + cell-width + array-pad-x
  let array-top = agent-top + array-pad-top
  let array-bottom = agent-bottom - array-pad-bottom

  rect(
    (array-left, array-top),
    (array-right, array-bottom),
    fill: array-fill,
    stroke: array-stroke,
    radius: 3pt,
  )

  content(
    (array-left + 1.0, array-top - 0.25),
    text(weight: "bold", size: 13pt, fill: array-stroke, "Agents[]"),
  )

  // Optional array brackets effect
  content(
    (array-left - 0.15, (array-top + array-bottom) / 2),
    text(size: 28pt, weight: "bold", fill: array-stroke, "["),
  )
  content(
    (array-right + 0.15, (array-top + array-bottom) / 2),
    text(size: 28pt, weight: "bold", fill: array-stroke, "]"),
  )

  // -----------------------------
  // Agents as array entries
  // -----------------------------
  for (j, agent) in agent-labels.enumerate() {
    let x = agent-x(j)

    if j == agent-labels.len() - 1 {
      let dots-x = x - ellipsis-gap / 2 - margin / 2
      content(
        (dots-x, -2.5),
        text(size: 25pt, weight: "bold", fill: panel-text, [$dots$]),
      )
      content(
        (dots-x, array-bottom - 0.25),
        text(size: 10pt, fill: panel-text, "…"),
      )
    }

    draw-agent-column(x, agent)

    let array-label = if j == 2 {
      "n -1"
    } else {
      str(j)
    }

    content(
      (x + cell-width / 2, array-bottom - 0.25),
      text(size: 10pt, fill: panel-text, "[" + array-label + "]"),
    )
  }

  // -----------------------------
  // Resources
  // -----------------------------
  let resources = (
    (0.5, -3 * cell-height, "Resource A", "Occupation Table"),
    (-3.0, -7.0 * cell-height, "Resource B", "RNG"),
  )

  for (top, bottom, title, subtitle) in resources {
    draw-resource-panel(resource-x, top, bottom, title, subtitle)
  }
})

#let ecs-data-layout = fletcher-diagram({
  let teal = rgb("#2f7c82")
  let teal-light = rgb("#d9eef1")
  let orange = rgb("#d97b34")

  let rows = (
    (label: "Position", present: (true, true, true)),
    (label: "Direction", present: (true, true, true)),
    (label: "Habitus", present: (true, true, false)),
    (label: "Convention", present: (true, false, true)),
    (label: "SocialHabit", present: (true, false, false)),
  )
  let entities = ("Car A", "Car B", "Car Z")
  let cell-xs = (3.4cm, 4.4cm, 5.4cm)

  // entity headers
  for (j, e) in entities.enumerate() {
    node((cell-xs.at(j), 5.7cm), text(fill: teal, weight: "bold", size: 9pt, e))
  }
  node((6.3cm, 5.7cm), text(fill: teal, size: 12pt, weight: "bold", [$dots.h$]))

  // component rows
  for (i, r) in rows.enumerate() {
    let y = (4 - i) * 1.15cm
    node(
      (3.15cm, y),
      align(left, text(fill: teal-light, weight: "bold", size: 9.5pt, r.label)),
      width: 5.7cm,
      height: 1.0cm,
      inset: 8pt,
      fill: teal,
      corner-radius: 4pt,
      name: label("row-" + str(i)),
    )

    for j in range(3) {
      if r.present.at(j) {
        node(
          (cell-xs.at(j), y),
          text(size: 7.5pt, fill: teal.darken(25%), style: "italic", "[data]"),
          width: 0.8cm,
          height: 0.6cm,
          fill: white,
          corner-radius: 3pt,
          stroke: none,
        )
      } else {
        node(
          (cell-xs.at(j), y),
          "",
          width: 0.8cm,
          height: 0.6cm,
          fill: none,
          corner-radius: 3pt,
          stroke: (paint: teal.lighten(45%), thickness: 0.8pt, dash: "dashed"),
        )
      }
    }
  }

  // systems
  node(
    (9.6cm, 4.0cm),
    align(center)[
      #text(fill: orange.darken(35%), weight: "bold", size: 10pt)[`move`] \
      #text(size: 7pt, fill: luma(70))[reads: Position, Direction]
    ],
    fill: orange.lighten(85%),
    stroke: 1pt + orange.darken(10%),
    corner-radius: 4pt,
    inset: 8pt,
    name: <s-move>,
  )
  node(
    (9.6cm, 2.3cm),
    align(center)[
      #text(fill: orange.darken(35%), weight: "bold", size: 10pt)[`update_habitus!`] \
      #text(size: 7pt, fill: luma(70))[reads: Position, Direction, Habitus]
    ],
    fill: orange.lighten(85%),
    stroke: 1pt + orange.darken(10%),
    corner-radius: 4pt,
    inset: 8pt,
    name: <s-hab>,
  )
  node(
    (9.6cm, 0.6cm),
    align(center)[
      #text(fill: orange.darken(35%), weight: "bold", size: 10pt)[`propose_lanes!`] \
      #text(size: 7pt, fill: luma(70))[reads: Position, Habitus, Convention, SocialHabit]
    ],
    fill: orange.lighten(85%),
    stroke: 1pt + orange.darken(10%),
    corner-radius: 4pt,
    inset: 8pt,
    name: <s-lanes>,
  )

  // arrows: system -> component row
  edge(<s-move>, <row-0>, "->", stroke: 1.1pt + luma(90))
  edge(<s-move>, <row-1>, "->", stroke: 1.1pt + luma(90))
  edge(<s-hab>, <row-2>, "->", stroke: 1.1pt + luma(90))
  edge(<s-lanes>, <row-2>, "->", stroke: 1.1pt + luma(90))
  edge(<s-lanes>, <row-3>, "->", stroke: 1.1pt + luma(90))
  edge(<s-lanes>, <row-4>, "->", stroke: 1.1pt + luma(90))
})
