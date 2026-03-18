#import "@preview/ilm:1.4.2": *

#set text(lang: "en")


#show: ilm.with(
  title: [Entity component systems - a novel way to do agent baasd modeling],
  author: ("Franz Scharnreitner BSc."),
  date: datetime(year: 2026, month: 02, day: 7),
  abstract: [In this paper],
  bibliography: bibliography("Econ.bib"),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true)
)

= Why Agent Based modeling?

Agent based modeling (ABM) has cemented itself one of the core approaches to modeling in the heterodox economic sphere, because of it's abbilty 
to capture complex system interaction as the interaction between individual agents whose behaviour can be described by manageable rulesets.


= Challenges of ABMs

There exists a multitude of different frameworks for different languages and with varying complexity and sophistication that make use of different techniques for implenting agency of individual agents @abarAgentBasedModelling2017. 
However most of these frameworks have commonality in the fact that they model agent behaviour as a function of the agent and the model state. This is mostly done in two different ways:
+ Via a step function that may be scheduled by the model at every time step.
+ Via a function that reacts on certain events.

The sequential nature of this imposes a crucial limitation on the modeling capabilities of agent based models: paralellism - both in the modeling sense and in the implementation - become nearly impossible.



= ECS  



== A simple SIR Model

== Revisiting a classic heterodox ABM
=== A true port
=== Towards a more natural ECS implentation










