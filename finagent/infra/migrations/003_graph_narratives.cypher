CREATE FULLTEXT INDEX market_narrative_fulltext IF NOT EXISTS
FOR (n:MarketNarrative) ON EACH [n.content, n.summary, n.name];

CALL db.awaitIndexes();
