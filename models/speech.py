import speech_recognition as sr
import os
import datetime
from utils.config import logger, TRANSCRIPTIONS_DIR

class SpeechHandler:
    """Classe para tratamento de reconhecimento de fala"""
    
    def __init__(self, sentiment_analyzer):
        self.recognizer = sr.Recognizer()
        self.sentiment_analyzer = sentiment_analyzer
        logger.info("SpeechHandler inicializado")
    
    def ouvir_microfone(self):
        """Captura áudio do microfone e retorna sua transcrição"""
        try:
            logger.info("Iniciando captura de áudio do microfone...")
            with sr.Microphone() as source:
                # Ajustar para ambiente
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Ajuste para ruído ambiente concluído. Aguardando fala...")
                
                # Capturar áudio
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)
                logger.info("Áudio capturado com sucesso! Transcrevendo...")
                
                try:
                    # Transcrever áudio capturado
                    texto = self.recognizer.recognize_google(audio, language='pt-BR')
                    logger.info(f"Transcrição obtida: '{texto[:50]}...'")
                    
                    # Analisar sentimento da transcrição
                    analise = self.sentiment_analyzer.analisar_sentimento(texto)
                    logger.info(f"Análise de sentimento: {analise['sentimento']} (confiança: {analise['confianca']:.2f})")
                    
                    # Salvar transcrição em arquivo para histórico
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    caminho_arquivo = os.path.join(TRANSCRIPTIONS_DIR, f"transcricao_{timestamp}.txt")
                    
                    with open(caminho_arquivo, 'w', encoding='utf-8') as arquivo:
                        arquivo.write(f"Data: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                        arquivo.write(f"Texto: {texto}\n")
                        arquivo.write(f"Sentimento: {analise['sentimento']}\n")
                        arquivo.write(f"Confiança: {analise['confianca']:.2f}\n")
                        if 'modelo' in analise:
                            arquivo.write(f"Modelo utilizado: {analise['modelo']}\n")
                    
                    logger.info(f"Transcrição salva em {caminho_arquivo}")
                    
                    return {
                        'sucesso': True,
                        'texto': texto,
                        'analise': analise,
                        'arquivo': caminho_arquivo
                    }
                except sr.UnknownValueError:
                    logger.warning("Fala não reconhecida")
                    return {
                        'sucesso': False,
                        'erro': 'Não foi possível reconhecer a fala',
                        'codigo_erro': 'UNKNOWN_VALUE'
                    }
                except sr.RequestError as e:
                    logger.error(f"Erro na requisição ao serviço de reconhecimento: {e}")
                    return {
                        'sucesso': False,
                        'erro': f'Erro no serviço de reconhecimento: {e}',
                        'codigo_erro': 'REQUEST_ERROR'
                    }
                except Exception as e:
                    logger.error(f"Erro ao processar transcrição: {e}")
                    return {
                        'sucesso': False,
                        'erro': f'Erro inesperado: {e}',
                        'codigo_erro': 'UNEXPECTED_ERROR'
                    }
        except Exception as e:
            logger.error(f"Erro ao inicializar microfone: {e}")
            return {
                'sucesso': False,
                'erro': f'Erro ao acessar o microfone: {e}',
                'codigo_erro': 'MIC_ERROR'
            } 