import os
import logging
import asyncio
import yfinance as yf
import flask
import threading
from app.trading import TradingBot, DataManager
from app.telegram_bot import TelegramBot
from app.model_trainer import ModelTrainer
import schedule
import pandas as pd
from app import create_app

# Desactivar las operaciones personalizadas de oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuración de logging mejorada
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('trading_bot.log'),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

logger.info("Iniciando el bot de trading...")

# Listas de símbolos (sin puntos suspensivos)
forex_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"]
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]  # Cambiado FB por META
crypto_symbols = ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD"]
additional_symbols = ["F", "GM", "TM", "HMC", "TSLA"]

symbols = stock_symbols + forex_symbols + crypto_symbols + additional_symbols

input_file = 'market_data.csv'
cleaned_file = 'market_data_cleaned_auto.csv'

if not os.path.exists(input_file):
    logger.info(f"{input_file} no encontrado. Descargando datos del mercado...")
    try:
        # Cambiado para descargar datos para cada símbolo individualmente desde el principio
        all_data = []
        for symbol in symbols:
            try:
                symbol_data = yf.download(symbol, start='2001-01-01', end='2024-01-01')
                symbol_data['Symbol'] = symbol
                all_data.append(symbol_data)
                logger.info(f"Datos descargados exitosamente para {symbol}")
            except Exception as e:
                logger.warning(f"Error al descargar datos para {symbol}: {e}")
        if all_data:
            combined_data = pd.concat(all_data)
            combined_data.to_csv(input_file)
            logger.info(f"Datos combinados guardados en {input_file}")
        else:
            logger.error("No se pudo descargar ningún dato. Verifique su conexión a internet y el paquete yfinance.")
            exit(1)
    except Exception as e:
        logger.error(f"Error al descargar datos del mercado: {e}")
        exit(1)
else:
    logger.info(f"{input_file} ya existe. Omitiendo la descarga.")

if os.path.exists(input_file):
    logger.info(f"{input_file} existe.")
else:
    logger.error(f"No se pudo crear {input_file}. Saliendo.")
    exit(1)

# Limpiar los datos
from data_cleaner import clean_data
try:
    clean_data(input_file, cleaned_file)
    logger.info(f"Datos limpiados y guardados en {cleaned_file}")
except Exception as e:
    logger.error(f"Error durante la limpieza de datos: {e}")
    logger.info("Intentando proceder con datos sin limpiar...")
    cleaned_file = input_file

# Inicializar componentes
try:
    data_manager = DataManager(symbols)
    trading_bot = TradingBot()
    model_trainer = ModelTrainer(trading_bot)
    telegram_bot = TelegramBot()
    app = create_app()
    logger.info("Todos los componentes inicializados con éxito")
except Exception as e:
    logger.error(f"Error al inicializar componentes: {e}")
    exit(1)

def run_flask_app():
    try:
        app.run(debug=False, use_reloader=False)  # Añadido use_reloader=False para evitar problemas con threading
    except Exception as e:
        logger.error(f"La aplicación Flask encontró un error: {e}")

async def main():
    start_message = "El bot de trading ha iniciado con éxito."
    logger.info(start_message)

    try:
        await telegram_bot.send_message(start_message)
        logger.info("Mensaje de inicio enviado con éxito a través de Telegram.")
    except Exception as e:
        logger.error(f"Error al enviar mensaje de inicio a través de Telegram: {e}")

    try:
        logger.info("Iniciando entrenamiento del modelo...")
        await model_trainer.train_all_models(data_manager)
    except Exception as e:
        logger.error(f"Error durante el entrenamiento del modelo: {e}")

    try:
        logger.info("Iniciando entrenamiento de aprendizaje por refuerzo...")
        await trading_bot.run_reinforcement_learning(cleaned_file)  # Cambiado a async
    except Exception as e:
        logger.error(f"Error durante el aprendizaje por refuerzo: {e}")

    try:
        logger.info("Iniciando actualización de datos y bot de trading...")
        await data_manager.start_data_update(interval_minutes=5)  # Cambiado a async
    except Exception as e:
        logger.error(f"Error al iniciar la actualización de datos: {e}")

    try:
        headlines = [
            "El mercado de valores se desploma en medio de la incertidumbre económica",
            "Las acciones tecnológicas repuntan tras sólidos informes de ganancias",
            "Los inversores son optimistas sobre la recuperación económica"
        ]
        sentiments = await trading_bot.run_sentiment_analysis(headlines)  # Cambiado a async
        logger.info(f"Sentimientos: {sentiments}")
    except Exception as e:
        logger.error(f"Error durante el análisis de sentimientos: {e}")

    try:
        await trading_bot.run_backtest(cleaned_file)  # Cambiado a async
    except Exception as e:
        logger.error(f"Error durante el backtesting: {e}")

    while True:
        try:
            await schedule.run_pending()  # Cambiado a async
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error en el bucle principal: {e}")
            await asyncio.sleep(60)  # Esperar un minuto antes de reintentar

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Error crítico en la ejecución principal: {e}")