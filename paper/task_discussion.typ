#set document(
  title: "Task Discussion",
  author: "Benjamin Yu Sheng Chang",
  date: datetime.today(),
)

#set page(margin: (top: 1in, bottom: 1in, left: 1in, right: 1in))
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set bibliography(style: "ieee")

#let cplus = "\u{2295}"
#let ket(x) = [|#x⟩]

#align(center)[
  #text(20pt, weight: "bold")[Task Discussion] \
  #v(1em)
  #text(
    [Benjamin Yu Sheng Chang \ benji.chang\@mail.utoronto.ca]
  )
  #v(1.5em)
]

=== Task II

Base on the AUC benchmark in @Komiske2019. Most neural network methods such as RNN and CNN, achieved around 0.87 to 0.90 AUC. Our GCN achieved around $0.87$ baseline AUC without fine tuning. This competitive result is due to the permutation invariant natural of the GCN model. GCN and DeepSet both are permutation invariant which perfectly suits the dataset. The construction of the graph is based on the KNN neighbors where k=7, so two particles are neighbors in the graph iff they are k-neighbors in the same group of particles. The edge_index is based on the rapidity and phi because these two contains the spatial information that respects the local geometry.

=== Task V


#bibliography("refs.bib")