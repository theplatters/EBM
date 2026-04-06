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

- Influential paper by #cite(<hodgsonEconomicsShadowsDarwin2006>, form: "prose") #pause
- #image("assets/roundabout.jpg", height: 70%)
== Overview
- Cars drive sequentially either clockwise or counterclockwise on a $100 times 2$ ring and decide in each step, whether to turn take the left or right lane based on the decision formula $ "LR"^n = w_s  S_s^n (2 s^n - 1) +  
        w_o  S_o^n  (2 o^n - 1) + 
        w_a  S_a^n  (c_r^n - c_l^n) + 
        w_h S_h^n h^n. $ where $s, o$ and $c_r$/$c_l$ are determined by the cars ahead of car $n$ and $h^n$ is the habit of the driver.
-  Driving on one lane shifts their habit towards that lane based on the update formula
  $ h^n = h^n_"prev" + "lane" / (K + t) $

== The real world

#speaker-note[
  + Ask why abs enforce sequential logic that is and that we have to go into more detail
]
- Real-world agents (especially cars) operate in parallel.
- Why did #cite(<hodgsonEconomicsShadowsDarwin2006>, form: "prose") choose a sequential model? #pause 
  - Because it is simple?
  - $arrow.dashed$ In the classic ABM framework the sequential approach is the most natural approach. #pause
  - But why does the sequential approach lend itself so well for these models?
= ABM layouts

== Traditional ABM layout
#align(center)[

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

]
#speaker-note[
  + Agents have step functions 
  + these step functions run sequentialy
]

== What if?

#speaker-note[
  + What if we group the data not by agents but by systems
  + Luckily  paradigm exists => ECS
]
#align(center)[
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
})]


= ECS (Entity Component System)!

== The history of ECS

- The origins of ECS reach back to 1959 where Ivan Sutherland pioneered it in a drawing program,one of the first graphical user interfaces  @sutherlandSketchpadManmachineGraphical2003. 
- Now ECS is primarily used in game development. (e.g Bevy Engine, Unity Dots, ...)
  #align(center)[

 #image("assets/sketchpad.png", height: 39%)
  ]
== A short introduction

- ECS is a way to separate data from functionality
- 
#pagebreak(weak: true)


== Data layout in a classic ABM


#cetz-canvas(length: 1.6cm, {
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
    text(size: 10pt, fill: panel-text, "[" +array-label + "]"),
    )
  }

  // -----------------------------
  // Resources
  // -----------------------------
  let resources = (
    (0.5, -3 * cell-height, "Resource A", "Occupation Table"),
    (-3.0, -7.0 * cell-height, "Resource B", "RNG"),
  )

  for ((top, bottom, title, subtitle)) in resources {
    draw-resource-panel(resource-x, top, bottom, title, subtitle)
  }
})
== ECS data layout

#figure()[
  #image("assets/ECS_Simple_Layout.svg", height: 80%)
]

== ABM in ECS terms
#table(
  columns: (auto, auto, auto, auto),
  inset: 7pt,
  align: horizon,
  [*Concept*],
  [*Description*],
  [*ABM Equivalent*],
  [*Role in Simulation*],

  [*Entity*],
  [Unique object in the world],
  [Agent],
  [Represents an individual actor in the simulation],

  [*Component*],
  [Data attached to an entity],
  [Agent attributes / state variables],
  [Stores properties such as position, preferences, or resources],

  [*System*],
  [Function operating on sets of components],
  [Part of the agent step function],
  [Implements simulation rules and updates state],
)
= Cars 2

== Reimplementing the model using ECS

- The natural way to implement the model in ECS is so that cars move simultaneously.
- This leads to a problem:
  - How does a car predict where the other cars are in the next step?
  - $arrow.dashed$ different prediction strategies.

== Prediction strategies

#table(
  columns: (auto, auto),
  inset: 7pt,
  align: horizon,
  [*Strategy*],
  [*Prediction*],

  [*Naive strategy*],
  [Other cars just stay on their lane],

  [*Switch strategy*],
  [Other cars switch their lane],

  [*Unsure strategy*],
  [Both lanes are equally likely],

  [*Per Entity Habitus*],
  [The car switches lanes based on its habitus],

  [*Mean Absolute Habitus*],
  [The car is more likely to stay on a lane the higher the mean absolute habitus is],

  [*Random strategy*],
  [The car gets assigned a random probability to switch lane],
)

== Which strategy works best

#figure()[
  #image("../../plots/mean_age.png", width: 75%)
]
== The formation of habit
#figure()[
  #image("../../plots/habitus.png", width: 75%)
]




#show: appendix

= Appendix

== Appendix

#bibliography("../Econ.bib")
