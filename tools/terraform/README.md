# Terraform Provisioning

Simple utilities to train models automatically on cloud provisioned instances.

## Introduction

We can use Terraform to help us automate the running and testing of different models. The basic need is the ability to run our code (the current workspace code, not a git commit hash) on a cloud instance, monitor the progress, download the results and the log files, and terminate the instance.

We use lambda labs for all of our training, so the basic terraform configuration will be based on that. We can add other providers as well.

## Appendix

The terraform script relies on the Lambda Labs provider from [here](https://github.com/elct9620/terraform-provider-lambdalabs).