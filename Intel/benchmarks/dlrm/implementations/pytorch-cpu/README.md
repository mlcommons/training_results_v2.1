## HW and SW requirements

### 1. HW requirements

| HW  | Configuration           |
| --- | ----------------------- |
| CPU | Intel Xeon (R) codenamed Sapphire Rapids  |
| DDR | 512G/socket @ 3200 MT/s |
| SSD | 1 SSD/Node @ >= 1T      |

### 2. SW requirements

| SW       | Version |
| -------- | ------- |
| GCC      | >= 11.1 |
| Binutils | >= 2.35 |

## Run DLRM

NOTE: This is a preview submission for DLRM training on upcoming Xeon processors codenamed Sapphire Rapids, and is based on internal SW stack with the following versions:

PyTorch: 6e11a1b38b11b7df74d3f551c18dc8d7cbd24c66

Intel Extension for PyTorch: 635d28a5e11e1116403c01bfff77a91eadbf86cf

torch-ccl: v1.12.0
