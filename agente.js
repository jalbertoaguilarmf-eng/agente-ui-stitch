import { StitchToolClient } from "@google/stitch-sdk";
import OpenAI from "openai";
import dotenv from "dotenv";

// Cargar las variables de entorno desde el archivo .env
dotenv.config();

// Inicializar el cliente de OpenAI configurado para OpenRouter
const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
});

const modelo = process.env.OPENROUTER_MODEL || "openrouter/free";
let stitchClient;

async function iniciarAgente() {
  console.log("Iniciando conexión con Stitch MCP...");

  if (
    !process.env.STITCH_API_KEY &&
    !(process.env.STITCH_ACCESS_TOKEN && (process.env.STITCH_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT))
  ) {
    throw new Error(
      "Faltan credenciales de Stitch. Agrega STITCH_API_KEY al archivo .env, o configura STITCH_ACCESS_TOKEN junto con STITCH_PROJECT_ID."
    );
  }

  stitchClient = new StitchToolClient();

  // 2. Leer y traducir las herramientas de MCP para OpenRouter
  const mcpTools = await stitchClient.listTools();
  const openRouterTools = mcpTools.tools.map(tool => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema
    }
  }));

  console.log(`Conexión exitosa. ${openRouterTools.length} herramientas disponibles.`);

  // Definir los mensajes iniciales con el System Prompt y el User Prompt
  const mensajes = [
    {
      role: "system",
      content: `Eres un Ingeniero Frontend y Experto en UX/UI.
Tu objetivo es traducir los requerimientos del usuario en interfaces funcionales usando las herramientas de Stitch.
Reglas:
1. Analiza el requerimiento y decide qué herramientas usar.
2. Si una herramienta falla, lee el error, ajusta los parámetros y vuelve a intentarlo.
3. Informa al usuario de tus acciones de forma concisa.`
    },
    {
      role: "user",
      content: "Crea un proyecto nuevo y genera una pantalla de inicio de sesión con un botón azul oscuro, el cual reciba un usuario y password, así como las opciones de registrarse, olvidé mi contraseña y recordar sesión siempre."
    }
  ];

  let tareaCompletada = false;

  // 3. El Bucle del Agente (Agentic Loop)
  while (!tareaCompletada) {
    const respuesta = await openai.chat.completions.create({
      model: modelo,
      messages: mensajes,
      tools: openRouterTools,
    });

    console.log(`Modelo utilizado: ${respuesta.model}`);
    const mensajeModelo = respuesta.choices[0].message;
    mensajes.push(mensajeModelo);

    // Si el modelo decide usar una herramienta de Stitch
    if (mensajeModelo.tool_calls && mensajeModelo.tool_calls.length > 0) {
      for (const toolCall of mensajeModelo.tool_calls) {
        console.log(`Acción del Agente -> Ejecutando: ${toolCall.function.name}`);
        try {
          const argumentos = JSON.parse(toolCall.function.arguments);
          
          // Ejecutar la acción en la máquina local
          const resultado = await stitchClient.callTool(
            toolCall.function.name,
            argumentos
          );

          // Devolver el resultado de la herramienta a la memoria del LLM
          mensajes.push({
            role: "tool",
            tool_call_id: toolCall.id,
            name: toolCall.function.name,
            content: JSON.stringify(resultado)
          });

          console.log(`Acción completada con éxito.`);
        } catch (error) {
          const mensajeError = error instanceof Error ? error.message : String(error);
          const errorAutenticacion =
            /api key not valid|valid api key/i.test(mensajeError) ||
            (typeof error === "object" && error !== null && "code" in error && error.code === 401);

          if (errorAutenticacion) {
            throw new Error(
              "Stitch rechazó la API key (401). Verifica que STITCH_API_KEY sea una clave válida de Google Stitch; OPENROUTER_API_KEY es distinta.",
              { cause: error }
            );
          }

          console.log(`Error en la herramienta: ${mensajeError}. Informando al agente para que corrija...`);
          mensajes.push({
            role: "tool",
            tool_call_id: toolCall.id,
            name: toolCall.function.name,
            content: `Error: ${mensajeError}`
          });
        }
      }
    } else {
      // Si ya no usa herramientas, significa que terminó el trabajo y nos responde con texto
      console.log(`\nAgente: ${mensajeModelo.content}`);
      tareaCompletada = true;
    }
  }
}

iniciarAgente()
  .catch((error) => {
    console.error("Error al iniciar el agente:", error);
    process.exitCode = 1;
  })
  .finally(async () => {
    if (stitchClient) {
      try {
        await stitchClient.close();
      } catch (error) {
        console.error("Error al cerrar la conexión con Stitch:", error);
        process.exitCode = 1;
      }
    }
  });