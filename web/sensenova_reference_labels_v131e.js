import { app } from "../../scripts/app.js";

const REFERENCE_INPUT_PATTERN = /^(?:Image-\d+|images\.image(?:_\d+)?)$/;
const INPUT_SLOT_TYPE = 1;

function referenceNumber(name) {
    const current = /^Image-(\d+)$/.exec(name);
    if (current) {
        return Number.parseInt(current[1], 10);
    }
    if (name === "images.image") {
        return 1;
    }
    const match = /^images\.image_(\d+)$/.exec(name);
    return match ? Number.parseInt(match[1], 10) : null;
}

function referenceLabel(number) {
    // Original (upstream T8mars): `参考图 ${number} (Image-${number})`
    return `Reference Image ${number} (Image-${number})`;
}

function setInputLabel(input, label) {
    input.label = label;
    // ComfyUI's Vue canvas currently renders localized_name, while the legacy
    // canvas renders label. Set both without touching the stable input name.
    input.localized_name = label;
}

function updateReferenceLabels(node) {
    const inputs = (node.inputs ?? []).filter((input) => REFERENCE_INPUT_PATTERN.test(input.name));

    inputs.forEach((input, index) => {
        const number = index + 1;
        setInputLabel(input, referenceLabel(number));
    });

    node.graph?.trigger("node:slot-label:changed", {
        nodeId: node.id,
        slotType: INPUT_SLOT_TYPE,
    });
    node.setDirtyCanvas?.(true, true);
}

function scheduleReferenceLabels(node) {
    requestAnimationFrame(() => updateReferenceLabels(node));
    // The Vue graph manager attaches its trigger listener after graph restore.
    // Repeat the notification briefly so restored workflows and newly-created
    // nodes both receive the final display labels.
    for (const delay of [50, 500, 2000]) {
        window.setTimeout(() => updateReferenceLabels(node), delay);
    }
}

function isReferenceNode(node) {
    return ["SenseNovaReferenceImage", "SenseNovaReferenceImageAdvanced"].includes(node.comfyClass)
        || ["SenseNovaReferenceImage", "SenseNovaReferenceImageAdvanced"].includes(node.type);
}

function installReferenceLabels(node) {
    if (!isReferenceNode(node) || node.__sensenovaReferenceLabelsInstalled) {
        return;
    }
    node.__sensenovaReferenceLabelsInstalled = true;

    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        const result = onConnectionsChange?.apply(this, arguments);
        scheduleReferenceLabels(this);
        return result;
    };

    scheduleReferenceLabels(node);
}

function migrateLegacyReferenceInputs(graphData) {
    for (const node of graphData?.nodes ?? []) {
        if (!["SenseNovaReferenceImage", "SenseNovaReferenceImageAdvanced"].includes(node.type)) {
            continue;
        }
        const numbers = (node.inputs ?? [])
            .filter((input) => input.link != null)
            .map((input) => referenceNumber(input.name))
            .filter(Number.isInteger);
        const useAdvanced = node.type === "SenseNovaReferenceImage"
            && numbers.some((number) => number > 2);
        if (useAdvanced) {
            node.type = "SenseNovaReferenceImageAdvanced";
            if (node.properties?.["Node name for S&R"] === "SenseNovaReferenceImage") {
                node.properties["Node name for S&R"] = "SenseNovaReferenceImageAdvanced";
            }
        }
        if (node.type === "SenseNovaReferenceImage" && !useAdvanced) {
            // Old Autogrow workflows serialize one empty trailing slot. Drop
            // unconnected slots above Image-2 so the fixed common node does
            // not inherit a confusing phantom Image-3 during graph restore.
            node.inputs = (node.inputs ?? []).filter((input) => {
                const number = referenceNumber(input.name);
                return number === null || number <= 2 || input.link != null;
            });
        }
        for (const input of node.inputs ?? []) {
            const number = referenceNumber(input.name);
            if (input.name.startsWith("images.") && number !== null) {
                input.name = `Image-${number}`;
            }
        }
    }
    for (const subgraph of graphData?.definitions?.subgraphs ?? []) {
        migrateLegacyReferenceInputs(subgraph);
    }
}

app.registerExtension({
    name: "t8star.sensenova.reference-labels",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["SenseNovaReferenceImage", "SenseNovaReferenceImageAdvanced"].includes(nodeData.name)) {
            return;
        }

        // addInput runs before the Vue canvas creates its slot state. Supplying
        // both display fields here also survives graph configuration, which
        // intentionally preserves localized_name from the node definition.
        const addInput = nodeType.prototype.addInput;
        nodeType.prototype.addInput = function (name, type, extraInfo) {
            const number = referenceNumber(name);
            const label = number === null ? null : referenceLabel(number);
            const options = label === null
                ? extraInfo
                : { ...(extraInfo ?? {}), label, localized_name: label };
            const input = addInput.call(this, name, type, options);
            if (label !== null) {
                setInputLabel(input, label);
            }
            return input;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            scheduleReferenceLabels(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            scheduleReferenceLabels(this);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            scheduleReferenceLabels(this);
            return result;
        };
    },
    beforeConfigureGraph(graphData) {
        migrateLegacyReferenceInputs(graphData);
    },
    nodeCreated(node) {
        installReferenceLabels(node);
    },
    loadedGraphNode(node) {
        installReferenceLabels(node);
        scheduleReferenceLabels(node);
    },
});
