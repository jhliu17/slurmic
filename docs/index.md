---
sd_hide_title: true
---

::::::{div} landing-title
:style: "padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #438ff9 0%, #05A 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));"

::::{grid}
:reverse:
:gutter: 2 3 3 3
:margin: 4 4 1 2

:::

:::{grid-item}
:columns: 12
:child-align: justify
:class: sd-text-white sd-fs-3

# 🌩️ slurmic

:::{grid-item}
:columns: 12
:child-align: justify
:class: sd-text-white sd-fs-5

A package designed to provide seamless Python function execution on Slurm.

```{button-ref} get_started
:ref-type: doc
:outline:
:color: white
:class: sd-px-4 sd-fs-5

Get Started
```

:::
::::

::::::

::::{grid} 1 2 2 3
:margin: 4 4 0 0
:gutter: 1

:::{grid-item-card} {octicon}`table` Seamless Execution
:link: tutorials/sequential_jobs
:link-type: doc

Execute Python functions on Slurm just like local functions.
:::

:::{grid-item-card} {octicon}`note` Sequential Jobs and Dependencies
:link: tutorials/sequential_jobs
:link-type: doc

Map sequential jobs and manage job dependencies.
:::

:::{grid-item-card} {octicon}`duplicate` Distributed Training
:link: tutorials/distributed_training
:link-type: doc

Launch distributed jobs in a flexible way.
:::

::::

## Documentation

```{toctree}
:caption: Tutorials

get_started
tutorials/sequential_jobs
tutorials/distributed_training
tutorials/sharp_bits
```

```{toctree}
:caption: API
:maxdepth: 2

api
```
