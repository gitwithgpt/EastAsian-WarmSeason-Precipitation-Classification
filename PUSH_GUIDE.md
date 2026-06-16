# 推送指南（Push Guide）

本文件说明如何将本地仓库推送到 GitHub 远程仓库。

---

## 前提条件

1. 已安装 Git：`git --version`
2. 已配置 Git 用户名和邮箱：
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```
3. 已配置 GitHub 身份验证（SSH 密钥或 Personal Access Token）

---

## 推送步骤

### 第 1 步：进入本地仓库目录

```bash
cd "E:\julei\EastAsian-WarmSeason-Precipitation-Classification"
```

### 第 2 步：初始化 Git 仓库（如尚未初始化）

```bash
git init
```

### 第 3 步：添加远程仓库地址

```bash
git remote add origin https://github.com/gitwithgpt/EastAsian-WarmSeason-Precipitation-Classification.git
```

> 如果之前已添加过远程仓库，可先用 `git remote remove origin` 删除旧的，再重新添加。

### 第 4 步：添加所有文件到暂存区

```bash
git add .
```

### 第 5 步：提交更改

```bash
git commit -m "Initial commit: manuscript, figures, code, and data for Atmospheric Research submission"
```

### 第 6 步：推送到 GitHub

```bash
git branch -M main
git push -u origin main
```

---

## 验证推送成功

1. 打开浏览器访问：https://github.com/gitwithgpt/EastAsian-WarmSeason-Precipitation-Classification
2. 确认所有文件和目录都已显示在页面上
3. 检查 README.md 是否正常渲染

---

## 后续更新（如需要修改）

```bash
git add .
git commit -m "Update: [修改说明]"
git push origin main
```

---

## 常见问题

### Q1: 推送时提示 "Repository not found"

确保远程仓库地址正确，且你对该仓库有写入权限。如果是私有仓库，需确认已登录。

### Q2: 推送时提示 "Permission denied"

- 如果使用 HTTPS，可能需要输入 GitHub 用户名和 Personal Access Token（不是密码）
- 如果使用 SSH，确保 SSH 密钥已添加到 GitHub 账户：`ssh -T git@github.com`

### Q3: 文件太大无法推送

GitHub 单文件限制为 100MB。本仓库最大的文件约 1.6MB，不会触发此限制。如果将来需要上传更大的文件（如原始 `.npz` 数据），建议使用 Git LFS 或外部存储（Zenodo、Figshare）。

---

## 数据可用性声明中应引用的链接

在论文的 **Data Availability Statement** 中，请使用以下链接：

```
https://github.com/gitwithgpt/EastAsian-WarmSeason-Precipitation-Classification
```

完整的 Data Availability Statement 文本已包含在 `manuscript/manuscript.md` 中。
