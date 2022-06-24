import pandas as pd # Librería para el tratamiento de ficheros
from time import sleep
from datetime import datetime # Librería para el tratamiento de fechas
import numpy as np  # Librería para simplificar en matrices los resultados computacionales
from pylsl import StreamInlet, resolve_byprop  # Librería para recibir los datos de la EEG
import utils  # Libreria de utilidades
import math
from tincan import Verb, Agent # Librería para el uso de Learning Locker

# Definimos la posición de cada tipo de onda
class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

# Función encargada de realizar la valoración
def Escenario():
    from tincan import RemoteLRS # Librería para la conexión con Learning Locker
    # Definición de valores para la conexión
    lrs = RemoteLRS(
        version='1.0.0',
        endpoint='', #Dirección IP de la máquina del LL
        auth='' # Token de autenticación
    )
    # Definición de los verbos que se van a utilizar
    verb = Verb(
        id='http://adlnet.gov/expapi/verbs/asked',
    )
    # Definición del actor (participante)
    actor = Agent(
        #name='', #nombre que ha realizado el ejercicio
        mbox='', #email que ha realizado el ejercicio
    )
    # Definición del rango de tiempo del ejercicio
    since = datetime.strptime('20220601_11:00:00', '%Y%m%d_%H:%M:%S')
    until = datetime.strptime('20220601_11:20:00', '%Y%m%d_%H:%M:%S')
    # Definición de la query (consulta) que se va a realizar.
    query = {
        "agent": actor,
        "verb": verb,
        "related_activities": True,
        "related_agents": True,
        "since": since,
        "until": until
    }
    # Guardamos la respuesta en una variable
    response = lrs.query_statements(query)

    if not response:
        raise ValueError("statements could not be queried")

    # Lectura del archivo CSV y escritura de cada columna como lista en variables
    f = open('DatosDiadema2.csv')
    f2 = open('Informe.txt', 'w')
    df = pd.read_csv('DatosDiadema2.csv', sep=';', encoding='latin-1', )
    tiempo = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f:%z') for x in df['Tiempo'].tolist() if type(x) == type('')]
    valor = df['Valor'].tolist()
    tipo = df['Tipo'].tolist()
    valormedio = df['ValorMedio'].tolist()

    # Nos quedamos con los valores quitando los Nan
    valormedio2 = [x for x in valormedio if math.isnan(x) == False]

    # Asignamos los valores de las medias
    mediaalpha = valormedio[0]
    mediabeta = valormedio2[1]
    mediadelta = valormedio[2]
    mediatheta = valormedio2[3]

    # Calculamos el número de pistas y el número de filas del archivo CSV
    contadorpistas = len(response.content.statements)
    longitudcsv = len(tiempo)

    # Doble bucle que se encarga de la comprobación de coincidencias de los tiempos entre pistas y mediciones
    for i in range(longitudcsv):
        for j in range(contadorpistas):
            auxtimestamp = str(response.content.statements[j].timestamp)
            auxtimestampcsv = str(tiempo[i])

            if auxtimestamp[0:19] == auxtimestampcsv[0:19]:

                # Media de los diez segundos previos para cada tipo de onda
                mediadiez = (valor[i] + valor[i-4] + valor[i-8] + valor[i-12] + valor[i-16] + valor[i-20] + valor[i-24] + valor[i-28] + valor[i-32] + valor[i-36] + valor[i-40]) / 11

                # Escritura de la valoración al pedir una pista para cada tipo de onda en un archivo de texto
                if str(tipo[i]) == 'Alpha Media:':
                    if mediaalpha < mediadiez:
                        mediaalpha2 = abs(((mediaalpha/mediadiez) * 100) - 100)
                        f2.write(
                            'La media de: ' + str(tipo[i]) + 'en los 10 segundos previos a tomar la pista ' + str(j+1) + ' era: ' + str(
                                mediadiez)
                            + ' siendo un ' + str(mediaalpha2) + '% mayor que la media de las mediciones del ejercicio que ha sido: ' + str(mediaalpha) + '\n')
                    else:
                        mediaalpha2 = abs((mediaalpha/mediadiez * 100) - 100)
                        f2.write(
                            'La media de: ' + str(tipo[i]) + 'en los 10 segundos previos a tomar la pista ' + str(j+1) + ' era: ' + str(
                                mediadiez)
                            + ' siendo un ' + str(mediaalpha2) + '% menor que la media de las mediciones del ejercicio que ha sido: ' + str(mediaalpha) + '\n')

                if str(tipo[i]) == 'Beta Total:':
                    if mediabeta < mediadiez:
                        mediabeta2 = abs((mediabeta/mediadiez * 100) - 100)
                        f2.write(
                            'La media de: ' + str(tipo[i]) + 'en los 10 segundos previos a tomar la pista ' + str(j+1) + ' era: ' + str(
                                mediadiez)
                            + ' siendo un ' + str(mediabeta2) + '% mayor que la media de las mediciones del ejercicio que ha sido: ' + str(mediabeta) + '\n')
                    else:

                        mediabeta2 = abs((mediabeta/mediadiez * 100) - 100)
                        f2.write(
                            'La media de: ' + str(tipo[i]) + 'en los 10 segundos previos a tomar la pista ' + str(j+1) + ' era: ' + str(
                                mediadiez)
                            + ' siendo un ' + str(mediabeta2) + '% menor que la media de las mediciones del ejercicio que ha sido: ' + str(mediabeta) + '\n')
                if str(tipo[i]) == 'Delta Total:':
                    if mediadelta < mediadiez:
                        mediadelta2 = abs((mediadelta / mediadiez * 100) - 100)
                        f2.write(
                            'La media de: ' + str(tipo[i]) + 'en los 10 segundos previos a tomar la pista ' + str(j+1) + ' era: ' + str(
                                mediadiez)
                            + ' siendo un ' + str(mediadelta2) + '% mayor que la media de las mediciones del ejercicio que ha sido: ' + str(mediadelta) + '\n')
                    else:
                        mediadelta2 = abs((mediadelta / mediadiez * 100) - 100)
                        f2.write(
                            'La media de: ' + str(tipo[i]) + 'en los 10 segundos previos a tomar la pista ' + str(j+1) + ' era: ' + str(
                                mediadiez)
                            + ' siendo un ' + str(mediadelta2) + '% menor que la media de las mediciones del ejercicio que ha sido: ' + str(mediadelta) + '\n')

                if str(tipo[i]) == 'Theta Total:':
                    if mediatheta < mediadiez:
                        mediatheta2 = abs((mediatheta/mediadiez * 100) - 100)
                        f2.write(
                            'La media de: ' + str(tipo[i]) + 'en los 10 segundos previos a tomar la pista ' + str(j+1) + ' era: ' + str(
                                mediadiez)
                            + ' siendo un ' + str(mediatheta2) + '% mayor que la media de las mediciones del ejercicio que ha sido: ' + str(mediatheta) + '\n')
                    else:
                        mediatheta2 = abs((mediatheta/mediadiez * 100) - 100)
                        f2.write(
                            'La media de: ' + str(tipo[i]) + 'en los 10 segundos previos a tomar la pista ' + str(j+1) + ' era: ' + str(
                                mediadiez)
                            + ' siendo un ' + str(mediatheta2) + '% menor que la media de las mediciones del ejercicio que ha sido: ' + str(mediatheta) + '\n')

    f.close()
    f2.close()

# Función encargada de recopilar y escribir los datos
def BCI():
    global filter_state
    """ Parámetros """
    # Modificar estos parámetros para variar la obtención de los datos

    # Tamaño del Buffer para la recolección de los datos
    BUFFER_LENGTH = 3

    # Cuántas "epoch" utilizamos
    EPOCH_LENGTH = 1

    # Tiempo entre "epoch"
    OVERLAP_LENGTH = 0.5

    SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

    # Canales (electrodos) usados
    # 0 = Oreja Izquierda, 1 = Frente Izquierda, 2 = Frente Derecha, 3 = Oreja Derecha
    INDEX_CHANNELS = [1, 2]
    """ 1. Conexión con la diadema """

    # Search for active LSL streams
    print('Buscando una diadema disponible...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('No se ha encontrado ninguna diadema disponible')


    print("Empezando a recopilar datos")
    # Aplicamos correción de tiempo
    inlet = StreamInlet(streams[0], max_chunklen=12)

    # Obtener información de la diadema
    info = inlet.info()

    # Definimos la frecuencia de recolección (Para este caso 256 siempre)
    fs = int(info.nominal_srate())

    """ 2. Inicializando Buffers """

    # Inicializamos los buffer con los datos del EEG en crudo
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Inicializamos el buffer de la diadema
    # Tendrán el siguiente orden: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))  # 9 epochs/5 segundos (buffer)

    buffers = [[eeg_buffer, eeg_buffer, eeg_buffer], [band_buffer, band_buffer, band_buffer]]

    """ 3. Obtención de los datos """

    # Podemos parar el bucle con Control + C o con el botón de stop
    print('Utiliza Control + C o el botón de Stop para parar la obtención de los datos')

    # Abrimos el archivo y escribimos la primera fila, esta será usada como cabecera
    f = open('DatosDiadema2.csv', 'w')
    f.write('Tiempo;Tipo;Valor;ValorMedio\n')

    # Definimos unas variables que nos ayudarán a la hora de calcular la media más adelante
    Aaux = 0
    Baux = 0
    Daux = 0
    Taux = 0
    contador = 0

    try:
        i = 0

        Up = False
        # El siguiente bucle es el encargado de adquirir los datos de la diadema
        while True:
            i += 1
            for index in range(len(INDEX_CHANNELS)):
                """ 3.1 Adquiriendo Datos """
                # Obtenemos los datos de la EEG
                eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(SHIFT_LENGTH * fs))

                # Nos quedamos solo con el canal que nos interesa
                ch_data = np.array(eeg_data)[:, INDEX_CHANNELS[index]]

                # Actualizamos el buffer con el EEG
                buffers[0][index], filter_state = utils.update_buffer(
                    buffers[0][index], ch_data, notch=True,
                    filter_state=filter_state)

                """ 3.2 Calculamos los valores de las bandas """
                # Obtenemos los valores nuevos del EEG
                data_epoch = utils.get_last_data(buffers[0][int(index)],
                                                 EPOCH_LENGTH * fs)

                # Calculamos los valores de las bandas
                band_powers = utils.compute_PSD(data_epoch, fs)
                buffers[1][index], _ = utils.update_buffer(buffers[1][index], np.asarray([band_powers]))

            # Definimos los valores de la fecha
            now2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            now = datetime.strptime(now2, '%Y-%m-%d %H:%M:%S.%f')

            # Imprimimos los valores de cada tipo de onda con el fin de un seguimiento durante el ejercicio
            a = str('Alpha:;' + str(buffers[1][1][1, Band.Alpha]))
            amedia = str('Alpha Media:;' + str(np.mean(buffers[1][1][1, Band.Alpha])))

            bizq = str('Beta Izquierda:;' + str(buffers[1][0][-1][Band.Beta]))
            bder = str('Beta Derecha:;' + str(buffers[1][1][-1][Band.Beta]))
            b = str('Beta Total:;' + str(buffers[1][0][-1][Band.Beta] + buffers[1][1][-1][Band.Beta]))

            dizq = str('Delta Izquierda:;' + str(buffers[1][0][-1][Band.Delta]))
            dder = str('Delta Derecha:;' + str(buffers[1][1][-1][Band.Delta]))
            d = str('Delta Total:;' + str(buffers[1][0][-1][Band.Delta] + buffers[1][1][-1][Band.Delta]))

            tizq = str('Theta Izquierda:;' + str(buffers[1][0][-1][Band.Theta]))
            tder = str('Theta Derecha:;' + str(buffers[1][1][-1][Band.Theta]))
            t = str('Theta Total:;' + str(buffers[1][0][-1][Band.Theta] + buffers[1][1][-1][Band.Theta]))

            # Escribimos los valores en un archivo de texto
            f.write(str(now) +':+00:00' + ';' + amedia + '\n')
            f.write(str(now) +':+00:00' + ';' + b + '\n')
            f.write(str(now) +':+00:00' + ';' + d + '\n')
            f.write(str(now) +':+00:00' + ';' + t + '\n')

            # Cálculos necesarios para la posterior obtención de la media
            Aaux = (Aaux + np.mean(buffers[1][1][1, Band.Alpha]))
            Baux = (Baux + buffers[1][0][1][Band.Beta] + buffers[1][1][1][Band.Beta])
            Daux = (Daux + buffers[1][0][1][Band.Delta] + buffers[1][1][1][Band.Delta])
            Taux = (Taux + buffers[1][0][1][Band.Theta] + buffers[1][1][1][Band.Theta])
            contador += 1


    except KeyboardInterrupt:
        # Escribimos la media al final del documento y establecemos la cabecera
        f.write(';Alpha media para la sesion:' + ';;' + str(Aaux/contador) + '\n')
        f.write(';Beta media para la sesion:' + ';;' + str(Baux/contador) + '\n')
        f.write(';Deta media para la sesion:' + ';;' + str(Daux/contador) + '\n')
        f.write(';Theta media para la sesion:' + ';;' + str(Taux/contador) + '\n')
        f.close()
        sleep(5)
        f = open('DatosDiadema2.csv')
        df = pd.read_csv('DatosDiadema2.csv', sep=';', index_col=0, encoding='latin-1')
        df.columns = df.iloc[0]
        df = df[1:]
        df.head()
        df.to_csv('DatosDiademaEstudio.csv', index=False)
        f.close()

        print('Terminado ! Cerrando...')

# Definición del main
def main():
    BCI()
    Escenario()

# Llamada al main
if __name__ == "__main__":
    main()
