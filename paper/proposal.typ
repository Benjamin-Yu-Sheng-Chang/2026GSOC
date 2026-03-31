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
  #v(1em)
  #text(
    [Benjamin Yu Sheng Chang \ benji.chang\@mail.utoronto.ca]
  )
  #v(1.5em)
]

This project aims to implement a Quantum Graph Neural Network (QGNN) leveraging the PennyLane framework to advance machine learning applications in high-energy physics. We will develop a hybrid         
  quantum-classical architecture that exploits the natural graph structure inherent in particle collision data, utilizing quantum circuits to encode and process graph features with potential computational
   advantages. The QGNN will be applied to a standard high-energy physics benchmark dataset (such as the LHC Olympics or a similar classification task) to evaluate its performance against state-of-the-art
   classical graph neural networks and traditional machine learning methods. Through comprehensive benchmarking analysis comparing accuracy, training efficiency, and scalability, this work will provide
  empirical insights into whether quantum advantage is achievable for physics-informed machine learning tasks and identify promising directions for practical quantum machine learning applications in      
  particle physics.