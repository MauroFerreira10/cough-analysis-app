# 🚀 Como Publicar no GitHub e Usar no Google Colab

## 📋 Passo a Passo

### 1️⃣ Criar Repositório no GitHub

1. **Acesse:** https://github.com/new

2. **Preencha:**
   - **Repository name:** `cough-analysis-app`
   - **Description:** `Sistema de análise de tosse para detecção de pneumonia e bronquite usando Machine Learning`
   - **Visibility:** ✅ Public (para usar no Colab gratuitamente)
   - ⚠️ **NÃO marque** "Initialize this repository with a README"

3. **Clique em:** "Create repository"

---

### 2️⃣ Conectar Seu Projeto Local ao GitHub

Copie e execute no terminal:

```bash
cd /Users/mac/cough-analysis-app

# Adiciona repositório remoto (SUBSTITUA SEU-USUARIO pelo seu username do GitHub)
git remote add origin https://github.com/SEU-USUARIO/cough-analysis-app.git

# Renomeia branch para main (padrão moderno)
git branch -M main

# Faz upload do código
git push -u origin main
```

**Exemplo:**
Se seu usuário for `joaosilva`, o comando seria:
```bash
git remote add origin https://github.com/joaosilva/cough-analysis-app.git
```

---

### 3️⃣ Usar no Google Colab

#### Opção A: Upload do Notebook

1. Acesse: https://colab.research.google.com/
2. Clique em "Upload"
3. Selecione: `/Users/mac/cough-analysis-app/Colab_Training.ipynb`
4. No notebook, edite a primeira célula:
   ```python
   # SUBSTITUA com sua URL
   !git clone https://github.com/SEU-USUARIO/cough-analysis-app.git
   ```

#### Opção B: Abrir Direto do GitHub

1. Primeiro, faça upload do notebook para o GitHub:
   ```bash
   cd /Users/mac/cough-analysis-app
   git add Colab_Training.ipynb
   git commit -m "Adiciona notebook do Google Colab"
   git push
   ```

2. Depois, acesse diretamente:
   ```
   https://colab.research.google.com/github/SEU-USUARIO/cough-analysis-app/blob/main/Colab_Training.ipynb
   ```

---

### 4️⃣ Executar no Colab

1. **Abra o notebook** no Google Colab
2. **Ative GPU:**
   - Menu: `Runtime` → `Change runtime type`
   - Hardware accelerator: `GPU (T4)`
   - Clique em `Save`

3. **Execute as células** (Shift+Enter ou clique no ▶️)

4. **Aguarde:**
   - Setup: ~2 minutos
   - Treinamento: 10-30 minutos (depende dos dados)

---

## 🔧 Comandos Úteis

### Atualizar Código no GitHub

Sempre que fizer mudanças:

```bash
cd /Users/mac/cough-analysis-app

# Vê o que mudou
git status

# Adiciona mudanças
git add .

# Commit com mensagem
git commit -m "Descrição das mudanças"

# Envia para GitHub
git push
```

### Verificar Repositório Remoto

```bash
git remote -v
```

Deve mostrar algo como:
```
origin  https://github.com/SEU-USUARIO/cough-analysis-app.git (fetch)
origin  https://github.com/SEU-USUARIO/cough-analysis-app.git (push)
```

---

## 📊 Estrutura no GitHub

Seu repositório terá:

```
cough-analysis-app/
├── 📄 README.md                    # Página principal
├── 📄 Colab_Training.ipynb         # Notebook do Colab
├── 📁 src/                         # Código Python
├── 📁 mobile/                      # App Flutter
├── 📁 docs/                        # Documentação
└── 📄 requirements.txt             # Dependências
```

---

## 🎯 URL Final do Notebook

Será algo como:

```
https://colab.research.google.com/github/SEU-USUARIO/cough-analysis-app/blob/main/Colab_Training.ipynb
```

**Compartilhe essa URL** com qualquer pessoa para que possa executar o projeto!

---

## ✅ Checklist

Marque conforme for fazendo:

- [ ] Criar repositório no GitHub
- [ ] Conectar repositório local
- [ ] Fazer primeiro push
- [ ] Upload do notebook para GitHub
- [ ] Abrir notebook no Colab
- [ ] Ativar GPU no Colab
- [ ] Executar primeira célula (clone)
- [ ] Executar todas as células
- [ ] Baixar modelo treinado

---

## 🆘 Problemas Comuns

### Erro: "Permission denied (publickey)"

**Solução:** Use HTTPS em vez de SSH:
```bash
git remote set-url origin https://github.com/SEU-USUARIO/cough-analysis-app.git
```

### Erro: "Repository not found"

**Solução:** Verifique se:
- Repositório foi criado no GitHub
- URL está correta (sem erros de digitação)
- Repositório é público

### Notebook não encontra arquivos

**Solução:** Verifique se executou a célula de clone:
```python
!git clone https://github.com/SEU-USUARIO/cough-analysis-app.git
%cd cough-analysis-app
```

---

## 🎓 Recursos Adicionais

### GitHub
- **Documentação:** https://docs.github.com/
- **Tutorial Git:** https://guides.github.com/

### Google Colab
- **Documentação:** https://colab.research.google.com/notebooks/
- **Dicas:** https://colab.research.google.com/notebooks/basic_features_overview.ipynb

---

## 📝 Exemplo Completo

```bash
# 1. Configure seu usuário Git (se ainda não fez)
git config --global user.name "Seu Nome"
git config --global user.email "seu.email@example.com"

# 2. No diretório do projeto
cd /Users/mac/cough-analysis-app

# 3. Adicione o repositório remoto (MUDE SEU-USUARIO)
git remote add origin https://github.com/SEU-USUARIO/cough-analysis-app.git

# 4. Envie para GitHub
git branch -M main
git push -u origin main

# 5. Se der erro de autenticação, GitHub pedirá:
# - Username: seu-usuario
# - Password: seu-token (não a senha! Use token de acesso pessoal)
#   Crie token em: https://github.com/settings/tokens
```

---

## 🎉 Pronto!

Agora seu projeto está no GitHub e pode ser executado no Google Colab por qualquer pessoa!

**URL para compartilhar:**
```
https://github.com/SEU-USUARIO/cough-analysis-app
```

**URL do Colab:**
```
https://colab.research.google.com/github/SEU-USUARIO/cough-analysis-app/blob/main/Colab_Training.ipynb
```

---

*Criado em: Dezembro 2025*  
*Licença: MIT*

