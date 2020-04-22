## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/brichardson36/ML-project-spring2020/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List



**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### LSTM Neural Networks
  - LSTM neural networks are a form of recurrent neural network.  Normal RNN's suffer from what is called short-term memory, where if a sequence is very long, information can be lost when carrying it from previous steps.  Layers that get a small gradient update stop learning, usually earlier layers, RNN’s can forget what it seen in longer sequences, thus having a short-term memory.  LSTM’s fix short-term memory by having internal mechanisms called gates that regulate the flow of information.  These gates can learn which data in a sequence is important to keep or throw away. Thus, it can pass relevant information down the long chain of sequences to make predictions.  A times series problem such as stock trading can occur over a very long segment of time and need a very long neural network to predict, so a LSTM is a very good approach to this problem.


### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/brichardson36/ML-project-spring2020/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
