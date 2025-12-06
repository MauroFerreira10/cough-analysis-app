# Guia de Contribuição

Obrigado por considerar contribuir para o projeto de Análise de Tosse! 🎉

## 🤝 Como Contribuir

### Reportando Bugs

Se encontrar um bug, abra uma issue com:
- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs. observado
- Screenshots (se aplicável)
- Informações do ambiente (OS, Python version, etc.)

### Sugerindo Melhorias

Para sugerir novas funcionalidades:
- Abra uma issue com tag `enhancement`
- Descreva a funcionalidade proposta
- Explique por que seria útil
- Forneça exemplos de uso

### Pull Requests

1. **Fork** o repositório
2. **Clone** seu fork: `git clone https://github.com/seu-usuario/cough-analysis-app.git`
3. **Crie uma branch**: `git checkout -b feature/minha-feature`
4. **Faça suas alterações**
5. **Teste** suas mudanças
6. **Commit**: `git commit -m "Adiciona nova feature"`
7. **Push**: `git push origin feature/minha-feature`
8. Abra um **Pull Request**

## 📝 Estilo de Código

### Python

- Siga [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) para formatação
- Máximo 100 caracteres por linha
- Docstrings no formato Google

Exemplo:
```python
def process_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Processa áudio para análise.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem em Hz
    
    Returns:
        Áudio processado
    """
    # Sua implementação aqui
    pass
```

### Flutter/Dart

- Siga [Effective Dart](https://dart.dev/guides/language/effective-dart)
- Use `flutter format` antes de commit
- Mantenha widgets pequenos e reutilizáveis

## 🧪 Testes

- Adicione testes para novas funcionalidades
- Mantenha cobertura de testes > 80%
- Execute `make test` antes de submeter PR

## 📚 Documentação

- Atualize README.md se necessário
- Documente funções públicas
- Adicione comentários para lógica complexa

## 🔄 Processo de Review

1. Pelo menos 1 revisor deve aprovar o PR
2. Todos os testes devem passar
3. Código deve seguir padrões de estilo
4. Documentação deve estar atualizada

## 📋 Checklist do PR

Antes de submeter:

- [ ] Código segue padrões de estilo
- [ ] Testes foram adicionados/atualizados
- [ ] Documentação foi atualizada
- [ ] Todos os testes passam
- [ ] Branch está atualizado com main
- [ ] Commits têm mensagens descritivas

## 🎯 Áreas de Contribuição

Estamos especialmente interessados em:

- **Novos datasets**: Integração com datasets adicionais
- **Melhorias no modelo**: Novas arquiteturas, técnicas
- **Pré-processamento**: Melhores técnicas de limpeza de áudio
- **UI/UX**: Melhorias na aplicação móvel
- **Documentação**: Tutoriais, exemplos
- **Testes**: Aumentar cobertura
- **Performance**: Otimizações

## 💬 Comunicação

- **Issues**: Para bugs e feature requests
- **Discussions**: Para perguntas e discussões gerais
- **Email**: [seu-email] para questões privadas

## 📄 Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a mesma licença do projeto (MIT).

## 🙏 Agradecimentos

Todas as contribuições, grandes ou pequenas, são valorizadas e apreciadas!

