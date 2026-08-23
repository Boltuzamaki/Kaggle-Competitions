// Common-random-numbers harness for LOCAL EVALUATION ONLY.
//
// The shipped engine API (Api.h ApiBattleStart) hardcodes config.deviceRand = true
// and reseeds the generator from std::random_device, so no two local games can
// ever be replayed identically. The engine itself fully supports determinism:
// GameConfig carries a `seed`, and `deviceRand` selects between that seeded
// mt19937 and a fresh random_device at every randomness site (deck shuffle in
// CardMove.h, coin flips in SelectProc.h, effects in EffectInstant.h).
//
// This file adds ONE new entry point, CrnBattleStart(cards, seed), which performs
// exactly the same deck validation as ApiBattleStart but runs with a caller-chosen
// seed and deviceRand = false. Everything else delegates to the engine's own
// inline Api* functions. No engine file is modified.
//
// Purpose: paired A/B evaluation. Running agent A and agent B over the SAME seeds
// removes shuffle/flip variance from the comparison, which is the dominant noise
// source in our arena results. This is a measurement instrument used offline; the
// submitted agent still runs against Kaggle's official binary, and nothing here
// depends on or exploits engine bugs.
//
// The engine source is licensed for competition use only and must not be shared.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "All.h"

#ifdef _MSC_VER
#	define GAME_API __declspec(dllexport)
#else
#	define GAME_API __attribute__ ((visibility("default")))
#endif

extern "C" {

  GAME_API void GameInitialize() {
    InitializeAll();
  }

  // Seeded, reproducible battle start. Mirrors ApiBattleStart's validation.
  GAME_API StartData CrnBattleStart(int* cards, unsigned int seed) {
    ApiData* data = new ApiData();
    data->apiDataType = 1;

    GameConfig config = {};
    config.seed = seed ? seed : 1u;   // 0 would make Game::init draw a random seed
    config.recordLog = true;
    config.deviceRand = false;        // route every RNG use through the seeded mt19937

    for (int i = 0; i < 2; i++) {
      std::unordered_map<std::u8string, int> nameCount;
      bool aceSpec = false;
      bool basic = false;
      for (int j = 0; j < DECK_SIZE; j++) {
        CardId id = cards[i * DECK_SIZE + j];
        if (!CardTable.contains(id)) {
          delete data;
          return { nullptr, i, 1 };
        }
        const CardMaster& master = CardTable.at(id);
        if (master.aceSpec) {
          if (aceSpec) { delete data; return { nullptr, i, 4 }; }
          aceSpec = true;
        }
        if (master.cardType == CardType::Pokemon &&
            master.evolutionType == EvolutionType::Basic) {
          basic = true;
        }
        int& count = nameCount[master.name];
        count++;
        if (count > DECK_SAME_CARD_MAX && master.cardType != CardType::BasicEnergy) {
          delete data;
          return { nullptr, i, 2 };
        }
        config.decks[i].cards[j] = cards[i * DECK_SIZE + j];
      }
      if (!basic) { delete data; return { nullptr, i, 3 }; }
    }

    data->init(config);
    // Deliberately NOT reseeding from random_device here -- that is the single
    // difference from ApiBattleStart, and the whole point of this harness.
    data->start();
    data->next();
    return { data, -1, 0 };
  }

  GAME_API void BattleFinish(ApiData* data) {
    return ApiBattleFinish(data);
  }

  GAME_API SerialData GetBattleData(ApiData* data) {
    if (data->apiDataType != 1) {
      return {};
    }
    if (data->preGetSelectCount != data->selectCount) {
      const State& state = data->state;
      int index = std::max(state.logIndex[0], state.logIndex[1]);
      const std::vector<int>* selected = nullptr;
      if (data->visData.size() > 0) {
        selected = &data->selected;
      }
      ToJsonVis(data->state, data->jsonBuilder, index, selected);
      data->visData.push_back(data->jsonBuilder.buf);
      data->preGetSelectCount = data->selectCount;
    }
    return ApiGetBattleData(data);
  }

  GAME_API int Select(ApiData* data, int* select, int selectCount) {
    if (data->apiDataType != 1) {
      return 30;
    }
    return ApiSelect(data, select, selectCount);
  }

  GAME_API const char8_t* AllCard() {
    static JsonBuilder b;
    if (b.buf.empty()) {
      ApiAllCard(b);
    }
    return b.buf.c_str();
  }

  GAME_API const char8_t* AllAttack() {
    static JsonBuilder b;
    if (b.buf.empty()) {
      ApiAllAttack(b);
    }
    return b.buf.c_str();
  }

}
