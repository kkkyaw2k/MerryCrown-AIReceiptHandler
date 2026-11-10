# LangGraph Migration Diagrams

## 1. Target graph (state machine)

```plantuml
@startuml
title MerryCrown Receipt Pipeline - LangGraph Target

state "group_images\n(Stage 1: Gemini Flash Lite)" as group_images
state fanout <<fork>>

[*] --> group_images : POST /receipt/upload\n(BatchState: request, img_urls)
group_images --> fanout : groups found\n(1 image skips LLM call)
fanout --> transcribe : Send(group 1)
fanout --> transcribe : Send(group 2..N)

state "transcribe\n(Stage 2: Gemini Flash Lite)" as transcribe
state is_receipt <<choice>>
transcribe --> is_receipt

is_receipt --> [*] : NOT_A_RECEIPT\n(skip branch)
is_receipt --> parse : markdown

state "parse\n(Stage 3: DeepSeek, structured output)" as parse
state "validate\nsum((price-discount)*qty) ~= total" as validate
state valid <<choice>>

parse --> validate
validate --> valid
valid --> parse : mismatch\n(retry once with error)
valid --> save_draft : OK

state "save_draft\n(Supabase draft_receipt,\nstatus=pending_review)" as save_draft
state join <<join>>
save_draft --> join
join --> [*] : reducer merges results

note right of parse
  Checkpointer: PostgresSaver
  thread_id = batch_id
  enables resume after restart
end note
@enduml
```

## 2. Sequence (request flow)

```plantuml
@startuml
title Request Flow

actor Client
participant "FastAPI\n/receipt/upload" as API
participant "LangGraph\n(graph.ainvoke)" as Graph
participant "OpenRouter\nLLMs" as LLM
database "Supabase\ndraft_receipt" as DB
database "Postgres\nCheckpointer" as CP

Client -> API : POST {batch_id, customer_id, img_urls}\nx-api-key
API --> Client : 202 {status: accepted}
API ->> Graph : BackgroundTask\nthread_id = batch_id

Graph -> LLM : Stage 1 group images
LLM --> Graph : [[url1,url2],[url3]]
Graph -> CP : checkpoint

par per receipt group
  Graph -> LLM : Stage 2 transcribe -> markdown
  alt NOT_A_RECEIPT
    Graph -> Graph : end branch
  else receipt
    Graph -> LLM : Stage 3 parse -> JSON
    Graph -> Graph : validate totals
    opt mismatch
      Graph -> LLM : retry parse with error
    end
    Graph -> DB : insert draft_receipt
    Graph -> CP : checkpoint
  end
end
@enduml
```

## 3. Current vs target components

```plantuml
@startuml
title Components

package "Current" {
  [main.py\nFastAPI] as m1
  [receiptpipeline.py\nasyncio.gather + openai SDK] as p1
  [stage_config.json] as c1
  m1 --> p1
  p1 --> c1
}

package "Target" {
  [main.py\nFastAPI] as m2
  [graph.py\nLangGraph StateGraph] as g2
  [ChatOpenAI\n(OpenRouter base_url)] as l2
  [PostgresSaver] as cp2
  [stage_config.json] as c2
  m2 --> g2
  g2 --> l2
  g2 --> cp2
  l2 --> c2
}
@enduml
```

> Render with the VS Code "PlantUML" extension, IntelliJ's PlantUML plugin, or paste each block into https://www.plantuml.com/plantuml.
