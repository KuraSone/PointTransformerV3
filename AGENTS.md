# AGENTS.md

## 提交规范

- 必须使用[约定式提交](https://www.conventionalcommits.org/zh-hans/)格式，**scope 和正文必填**，footer 可选：

  ```
  <type>(<scope>): <description>

  <body>

  [optional footer(s)]
  ```

  示例：

  ```
  fix(deps): 将本地 wheels 索引设为 explicit

  避免依赖解析时误用本地索引。
  ```

- 提交消息（标题和正文）必须使用中文；type 和 scope 用英文。
- 每次提交时，必须把所有未推送的提交合并成一个提交：

  ```bash
  git log @{u}..HEAD --oneline   # 查看未推送的提交
  git reset --soft @{u}          # 撤销这些提交，保留改动
  git add <文件>
  git commit                     # 写一条涵盖全部改动的提交消息
  ```

  如果没有未推送的提交，就直接提交。

  如果当前分支没有上游分支（`@{u}` 报错，例如分支还没推送过），就以远程主分支为基准合并（`<remote>` 用 `git remote` 查，不一定是 `origin`）：

  ```bash
  git fetch <remote>
  git log <remote>/main..HEAD --oneline              # 查看未推送的提交
  git reset --soft $(git merge-base HEAD <remote>/main)
  git add <文件>
  git commit
  ```

  如果连远程仓库都没有，就不要合并，直接提交，并询问用户。

- 用户没有明确要求推送时，禁止自行推送。
