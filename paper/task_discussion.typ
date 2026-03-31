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

Based on the AUC benchmark in @Komiske2019, most neural network methods such as RNN and CNN achieved around 0.87 to 0.90 AUC. Our GCN achieved around $0.87$ baseline AUC without fine-tuning. This competitive result is due to the permutation invariant nature of the GCN model. Both GCN and DeepSet are permutation invariant, which perfectly suits the dataset. The graph construction is based on k-nearest neighbors where k=7, so two particles are neighbors in the graph if and only if they are k-neighbors within the same particle group. The edge index is based on rapidity and phi because these two coordinates contain spatial information that respects local geometry. The other model uses PMLP in @Yang2023, which is simpler it relies solely on MLPs in training and simple graph aggregation in inference.The metric is not as competitive as the GCN because the model complexity is much simpler, but it's a good baseline for graph neural network in general. 

=== Task V


#bibliography("refs.bib")