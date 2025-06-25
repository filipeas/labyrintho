#!/bin/bash

echo ""
echo "⚠️  Nota: Se você executou este script com './setup.sh', o ambiente não permanecerá ativo após o script terminar."
echo "👉 Ao finalizar a execução do script, rode o comando sugerido para ativar o ambiente."
echo ""

echo "Escolha uma das opções:"
echo "1) Criar ambiente virtual local (venv)"
echo "2) Criar ambiente Conda"
echo "3) Usar ambiente Conda existente"
read -p "Digite o número da opção desejada: " option


# Inicializa o conda para o shell
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "miniconda3 encontrado."
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "anaconda3 encontrado."
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "❌ Não foi possível encontrar conda.sh. Verifique sua instalação do Conda."
    exit 1
fi


# Função para instalar os requisitos
install_requirements() {
    echo "Instalando dependências do requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
}

if [ "$option" == "1" ]; then
    read -p "Digite o nome do ambiente local que deseja criar: " env_name

    # Remove se já existir
    rm -rf "$env_name"
    echo "Criando ambiente virtual local '$env_name'..."
    python -m venv "$env_name"
    source "$env_name/bin/activate"

    install_requirements

    echo "✅ Ambiente local '$env_name' criado e ativado!"
    echo "ℹ️ Para ativar este ambiente no futuro, execute:"
    echo "👉 source $env_name/bin/activate"

elif [ "$option" == "2" ]; then
    read -p "Digite o nome do novo ambiente Conda: " env_name

    echo "Criando ambiente Conda '$env_name'..."
    conda create -y -n "$env_name" python=3.10
    conda activate "$env_name"

    install_requirements

    echo "✅ Ambiente Conda '$env_name' criado e ativado!"
    echo "ℹ️ Para ativar este ambiente no futuro, execute:"
    echo "👉 conda activate $env_name"

elif [ "$option" == "3" ]; then
    echo "Ambientes Conda disponíveis:"
    conda env list
    read -p "Digite o nome do ambiente Conda existente que deseja usar: " env_name

    # Verifica se o ambiente existe
    if conda env list | grep -qE "^$env_name\s"; then
        echo "Ativando ambiente Conda '$env_name'..."
        conda activate "$env_name"

        install_requirements

        echo "✅ Ambiente Conda '$env_name' ativado!"
        echo "ℹ️ Para ativar este ambiente no futuro, execute:"
        echo "👉 conda activate $env_name"
    else
        echo "❌ Ambiente '$env_name' não encontrado. Abortando."
        exit 1
    fi

else
    echo "❌ Opção inválida. Abortando."
    exit 1
fi

echo "✅ Finished. Go Work!"
