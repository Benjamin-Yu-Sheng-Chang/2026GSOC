#set document(
  title: "2026 GSOC Proposal",
  author: "Benjamin Yu Sheng Chang",
  date: datetime.today(),
)

#set page(margin: (top: 1in, bottom: 1in, left: 1in, right: 1in))
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)

#let cplus = "\u{2295}"
#let ket(x) = [|#x⟩]

#align(center)[
  #text(20pt, weight: "bold")[2026 GSOC Proposal] \
  #text(16pt)[Quantum Graph Neural Networks for High
Energy Physics Analysis at the LHC] \
  #v(1em)
  #text(
    [Benjamin Yu Sheng Chang \ benji.chang\@mail.utoronto.ca]
  )
  #v(1.5em)
]


== Personal Information

- Full Name: Benjamin Yu Sheng Chang
- Location & Time Zone: UK, GMT+1
- Email: yusheng3077\@gmail.com
- GitHub Username: Benjamin-Yu-Sheng-Chang

== Background

=== Educational Background
4th year CS and Math Co-op student at University of Toronto.

=== Current Status
Research Assistant at Acceleration Consortium, an AI lab focused on self-driving chemical discovery. My role involves researching models that predict chemical properties and integrating them with custom Bayesian optimization loops.

=== GSoC Experience
First-time GSoC participant.

=== Relevant Experience & Skills
- Currently taking a Quantum Algorithms course at University of Toronto
- Strong track record of translating theoretical concepts into production applications
- Made a PR to fix dependency issues in a popular database library plugin
- Proficient in Python and TypeScript (3+ years experience)
- Familiar with ML frameworks (PyTorch, Pennylane) and HEP analysis tools

== Motivation

=== Why GSoC and QC-Devs?

I have benefited greatly from open-source projects in both my work and personal projects. I want to contribute to the open-source community to provide similar value to others. QC-Devs' focus on accessible quantum computing aligns perfectly with my interests and career goals.

=== Why This Project?

This project combines quantum computing and machine learning—two areas I am deeply passionate about. It bridges theory and practice, allowing me to implement cutting-edge research into a widely-used library that will benefit the quantum machine learning community.

=== Learning Goals

I want to learn the design patterns and abstractions used in production libraries to implement theoretical concepts. I have used multiple popular libraries and am fascinated by how they abstract complex algorithms to support multiple implementations elegantly. This project will be my first large-scale open-source contribution, and I am eager to learn professional development practices and collaborative workflows.

== Project Description

The ambitious HL-LHC program will require enormous computing resources in the next two decades. New technologies are being sought to
replace the present computing infrastructure. A burning question is whether quantum computers can solve the ever-growing demand for
computing resources in High-Energy Physics (HEP) in general and physics at LHC in particular. Discovery of new physics requires the
identification of rare signals against immense backgrounds. The development of machine learning methods will greatly enhance our ability to
achieve this objective. With this project we seek to implement Quantum Machine Learning methods for LHC HEP analysis based on the
Pennylane framework. This will enhance the ability of the HEP community to use Quantum Machine Learning methods.


== Project Goals

The primary objective of this project is to implement a QML model leveraging the PennyLane framework. We aim to:

- Implement a Quantum Graph Neural Network (QGNN) method based on a suitable framework e.g. Pennylane.
- Apply the quantum machine learning method to a benchmark high-energy physics analysis and benchmark the quantum machine learning performance compared to classical machine learning methods

== Expected Outcomes
- Trained Quantum Graph Neural Network with e.g. Pennylane framework.
- Apply the Quantum Machine Learning method to LHC physics analysis and compare to classical machine learning methods.